import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


CLASS_NAMES = ("Focused", "Mind-Wandering", "Drowsy")
UNCERTAIN_CLASS = "Data Uncertain"


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings for temporal tokens."""

    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even.")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        cos = self.cos[:seq_len].repeat_interleave(2, dim=-1).to(dtype=x.dtype, device=x.device)
        sin = self.sin[:seq_len].repeat_interleave(2, dim=-1).to(dtype=x.dtype, device=x.device)
        return (x * cos.unsqueeze(0)) + (self._rotate_half(x) * sin.unsqueeze(0))


class BlinkFeatureExtractor(nn.Module):
    """1D-CNN blink encoder used by the temporal model."""

    def __init__(self, input_channels: int = 1, output_dim: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 24, kernel_size=5, padding=2),
            nn.BatchNorm1d(24),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, output_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, blink_seq: torch.Tensor) -> torch.Tensor:
        if blink_seq.dim() != 3:
            raise ValueError("blink_seq must have shape (batch, seq, channels).")
        return self.network(blink_seq.transpose(1, 2)).squeeze(-1)


class MissingnessHandler(nn.Module):
    """Zero-mask or mean-impute missing stream samples."""

    def __init__(self, strategy: str = "mean"):
        super().__init__()
        if strategy not in {"zero", "mean"}:
            raise ValueError("strategy must be 'zero' or 'mean'.")
        self.strategy = strategy

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if valid_mask is None:
            return torch.nan_to_num(x)
        valid = valid_mask.unsqueeze(-1).to(device=x.device, dtype=torch.bool)
        x = torch.nan_to_num(x)
        if self.strategy == "zero":
            fill = torch.zeros_like(x)
        else:
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1)
            fill = (x * valid.to(x.dtype)).sum(dim=1, keepdim=True) / denom.to(x.dtype)
            fill = fill.expand_as(x)
        return torch.where(valid, x, fill)


class CrossAttentionFusion(nn.Module):
    """Bidirectional gaze/AU cross-attention with reliability gating."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gaze_to_au = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.au_to_gaze = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.reliability_gate = nn.Sequential(
            nn.Linear(dim * 2 + 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        gaze: torch.Tensor,
        au: torch.Tensor,
        gaze_valid: Optional[torch.Tensor] = None,
        au_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        au_padding = None if au_valid is None else ~au_valid
        gaze_padding = None if gaze_valid is None else ~gaze_valid
        gaze_context, _ = self.gaze_to_au(gaze, au, au, key_padding_mask=au_padding)
        au_context, _ = self.au_to_gaze(au, gaze, gaze, key_padding_mask=gaze_padding)

        gaze_quality = torch.ones(gaze.size(0), 1, device=gaze.device)
        au_quality = torch.ones(au.size(0), 1, device=au.device)
        if gaze_valid is not None:
            gaze_quality = gaze_valid.float().mean(dim=1, keepdim=True)
        if au_valid is not None:
            au_quality = au_valid.float().mean(dim=1, keepdim=True)

        pooled = torch.cat(
            [
                gaze_context.mean(dim=1),
                au_context.mean(dim=1),
                gaze_quality,
                au_quality,
            ],
            dim=-1,
        )
        weights = F.softmax(self.reliability_gate(pooled), dim=-1)
        fused = (weights[:, 0, None, None] * gaze_context) + (weights[:, 1, None, None] * au_context)
        return self.norm(gaze + self.dropout(fused)), weights


class TemporalTransformerEncoder(nn.Module):
    """Transformer backbone optimized for 150-frame windows."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_mult: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.rope = RotaryPositionalEmbedding(dim, max_seq_len=max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.rope(x)
        key_padding_mask = None if valid_mask is None else ~valid_mask
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class MultiModalTransformer(nn.Module):
    """Cross-attention MMT for Focused, Mind-Wandering, and Drowsy states."""

    def __init__(
        self,
        gaze_dim: int = 3,
        au_dim: int = 17,
        blink_dim: int = 1,
        model_dim: int = 128,
        blink_feature_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        confidence_threshold: float = 0.6,
        missing_strategy: str = "mean",
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.class_names = CLASS_NAMES
        self.missing = MissingnessHandler(missing_strategy)
        self.blink_extractor = BlinkFeatureExtractor(blink_dim, blink_feature_dim)
        self.gaze_projection = nn.Sequential(nn.Linear(gaze_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU())
        self.au_projection = nn.Sequential(nn.Linear(au_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU())
        self.blink_projection = nn.Sequential(nn.Linear(blink_feature_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU())
        self.cross_attention = CrossAttentionFusion(model_dim, num_heads=num_heads)
        self.modality_projection = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.temporal_encoder = TemporalTransformerEncoder(
            dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim // 2, len(CLASS_NAMES)),
        )

    def forward(
        self,
        gaze_seq: torch.Tensor,
        blink_seq: torch.Tensor,
        au_seq: torch.Tensor,
        gaze_valid: Optional[torch.Tensor] = None,
        au_valid: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if valid_mask is None:
            valid_mask = torch.ones(gaze_seq.shape[:2], dtype=torch.bool, device=gaze_seq.device)
        if gaze_valid is None:
            gaze_valid = valid_mask
        if au_valid is None:
            au_valid = valid_mask

        gaze_seq = self.missing(gaze_seq, gaze_valid)
        au_seq = self.missing(au_seq, au_valid)
        blink_seq = self.missing(blink_seq, valid_mask)

        gaze_tokens = self.gaze_projection(gaze_seq)
        au_tokens = self.au_projection(au_seq)
        fused_tokens, fusion_weights = self.cross_attention(gaze_tokens, au_tokens, gaze_valid, au_valid)
        blink_features = self.blink_extractor(blink_seq).unsqueeze(1).expand(-1, gaze_seq.size(1), -1)
        blink_tokens = self.blink_projection(blink_features)

        tokens = self.modality_projection(torch.cat([fused_tokens, blink_tokens], dim=-1))
        encoded = self.temporal_encoder(tokens, valid_mask=valid_mask)
        weights = valid_mask.float().unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(pooled)
        probabilities = F.softmax(logits, dim=-1)
        confidence, predictions = probabilities.max(dim=-1)
        is_uncertain = confidence < self.confidence_threshold
        display_predictions = torch.where(
            is_uncertain,
            torch.full_like(predictions, len(CLASS_NAMES)),
            predictions,
        )
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": display_predictions,
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "fusion_weights": fusion_weights,
        }

    def export_to_onnx(self, output_path: str, seq_len: int = 150) -> None:
        sample = (
            torch.zeros(1, seq_len, 3),
            torch.zeros(1, seq_len, 1),
            torch.zeros(1, seq_len, 17),
            torch.ones(1, seq_len, dtype=torch.bool),
            torch.ones(1, seq_len, dtype=torch.bool),
            torch.ones(1, seq_len, dtype=torch.bool),
        )
        torch.onnx.export(
            self.eval(),
            sample,
            output_path,
            input_names=["gaze_seq", "blink_seq", "au_seq", "gaze_valid", "au_valid", "valid_mask"],
            output_names=["logits", "probabilities", "predictions", "confidence", "is_uncertain", "fusion_weights"],
            dynamic_axes={
                "gaze_seq": {0: "batch", 1: "seq"},
                "blink_seq": {0: "batch", 1: "seq"},
                "au_seq": {0: "batch", 1: "seq"},
            },
            opset_version=17,
        )


@dataclass
class SubjectBaselineCalibration:
    """Collects the first seconds of a session and normalizes subject-specific signals."""

    fps: int = 30
    duration_seconds: int = 10
    gaze_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    blink_mean: float = 0.0
    blink_std: float = 1.0
    au_mean: np.ndarray = field(default_factory=lambda: np.zeros(17, dtype=np.float32))
    calibrated: bool = False
    _gaze: list = field(default_factory=list)
    _blink: list = field(default_factory=list)
    _au: list = field(default_factory=list)

    @property
    def required_frames(self) -> int:
        return self.fps * self.duration_seconds

    @property
    def progress(self) -> float:
        return min(1.0, len(self._gaze) / float(self.required_frames))

    def update(self, gaze: np.ndarray, blink: np.ndarray, au: np.ndarray) -> None:
        if self.calibrated:
            return
        self._gaze.append(np.asarray(gaze, dtype=np.float32))
        self._blink.append(float(np.asarray(blink).reshape(-1)[0]))
        self._au.append(np.asarray(au, dtype=np.float32))
        if len(self._gaze) >= self.required_frames:
            self.finalize()

    def finalize(self) -> None:
        if not self._gaze:
            return
        self.gaze_origin = np.nanmean(np.vstack(self._gaze), axis=0).astype(np.float32)
        blink = np.asarray(self._blink, dtype=np.float32)
        self.blink_mean = float(np.nanmean(blink))
        self.blink_std = float(max(np.nanstd(blink), 1e-3))
        self.au_mean = np.nanmean(np.vstack(self._au), axis=0).astype(np.float32)
        self.calibrated = True
        logger.info("Subject baseline calibration complete.")

    def transform(self, gaze: np.ndarray, blink: np.ndarray, au: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gaze_out = np.asarray(gaze, dtype=np.float32) - self.gaze_origin
        blink_out = (np.asarray(blink, dtype=np.float32) - self.blink_mean) / self.blink_std
        au_out = np.asarray(au, dtype=np.float32) - self.au_mean
        return gaze_out, blink_out, au_out


class StreamProcessor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GazeProcessor(StreamProcessor):
    pass


class BlinkProcessor(StreamProcessor):
    def __init__(self):
        super().__init__()
        self.extractor = BlinkFeatureExtractor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)


class FacialAUProcessor(StreamProcessor):
    def __init__(self, au_dim: int = 17):
        super().__init__()
        self.norm = nn.LayerNorm(au_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class EEGProcessor(StreamProcessor):
    pass


class HeartRateProcessor(StreamProcessor):
    pass


class InputStreamFactory:
    """Factory for adding input streams without changing the model core."""

    _registry = {
        "gaze": GazeProcessor,
        "blink": BlinkProcessor,
        "facial_au": FacialAUProcessor,
        "eeg": EEGProcessor,
        "heart_rate": HeartRateProcessor,
    }

    @classmethod
    def register(cls, stream_type: str, processor_cls: Any) -> None:
        cls._registry[stream_type] = processor_cls

    @classmethod
    def create_stream_processor(cls, stream_type: str, **kwargs: Any) -> StreamProcessor:
        if stream_type not in cls._registry:
            raise ValueError(f"Unknown stream type: {stream_type}")
        return cls._registry[stream_type](**kwargs)


class TorchInferenceBackend:
    """Low-latency PyTorch backend with CPU/GPU autocasting where available."""

    def __init__(self, model: Optional[MultiModalTransformer] = None, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or MultiModalTransformer()
        if model_path:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        # Keep the default backend conservative. TensorRT/OpenVINO/ONNX export is
        # available through export_to_onnx, while eager PyTorch avoids optional
        # Triton failures on standard Windows CPU/GPU setups.

    def predict(
        self,
        gaze: np.ndarray,
        blink: np.ndarray,
        au: np.ndarray,
        gaze_valid: Optional[np.ndarray] = None,
        au_valid: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        with torch.inference_mode():
            gaze_t = torch.as_tensor(gaze, dtype=torch.float32, device=self.device).unsqueeze(0)
            blink_t = torch.as_tensor(blink, dtype=torch.float32, device=self.device).unsqueeze(0)
            au_t = torch.as_tensor(au, dtype=torch.float32, device=self.device).unsqueeze(0)
            valid_t = torch.ones(gaze_t.shape[:2], dtype=torch.bool, device=self.device)
            gaze_valid_t = valid_t if gaze_valid is None else torch.as_tensor(gaze_valid, dtype=torch.bool, device=self.device).unsqueeze(0)
            au_valid_t = valid_t if au_valid is None else torch.as_tensor(au_valid, dtype=torch.bool, device=self.device).unsqueeze(0)
            out = self.model(gaze_t, blink_t, au_t, gaze_valid_t, au_valid_t, valid_t)
        prediction = int(out["predictions"].cpu().numpy()[0])
        state = UNCERTAIN_CLASS if prediction >= len(CLASS_NAMES) else CLASS_NAMES[prediction]
        return {
            "state": state,
            "confidence": float(out["confidence"].cpu().numpy()[0]),
            "probabilities": out["probabilities"].cpu().numpy()[0],
            "is_uncertain": bool(out["is_uncertain"].cpu().numpy()[0]),
            "fusion_weights": out["fusion_weights"].cpu().numpy()[0],
        }


# Backwards-compatible names for the original prototype tests.
EnhancedBlinkFeatureExtractor = BlinkFeatureExtractor


class MultiModalAttentionDetector(MultiModalTransformer):
    def __init__(self, au_dim: int = 17, temporal_model: str = "transformer", **kwargs: Any):
        super().__init__(au_dim=au_dim, **kwargs)
        self.temporal_model_name = temporal_model

    def forward(self, gaze_seq: torch.Tensor, blink_seq: torch.Tensor, au_seq: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return super().forward(gaze_seq, blink_seq, au_seq, *args, **kwargs)["logits"]


if __name__ == "__main__":
    batch, seq = 2, 150
    model = MultiModalTransformer()
    outputs = model(
        torch.randn(batch, seq, 3),
        torch.randn(batch, seq, 1),
        torch.randn(batch, seq, 17),
    )
    print({key: tuple(value.shape) if torch.is_tensor(value) else value for key, value in outputs.items()})
