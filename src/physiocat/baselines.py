from __future__ import annotations

import numpy as np
import torch
from scipy import signal
from torch import nn
from torch.nn import functional as F


def _sqi_summary(sqi: torch.Tensor, tokens: int) -> torch.Tensor:
    if sqi.ndim == 2:
        return sqi
    if sqi.ndim != 3 or sqi.shape[1] != 2:
        raise ValueError("sqi must have shape [batch,2] or [batch,2,tokens]")
    if sqi.shape[-1] != tokens:
        sqi = F.interpolate(sqi, size=tokens, mode="linear", align_corners=False)
    return sqi.mean(dim=-1)


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        bottleneck = max(8, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, 1),
            nn.GELU(),
            nn.Conv1d(bottleneck, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ResidualTemporalBlock(nn.Module):
    """Pre-activation residual temporal block with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dilation: int = 1, dropout: float = 0.10):
        super().__init__()
        padding = 2 * dilation
        self.main = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            SqueezeExcite1d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.main(x)


class MultiScaleStem(nn.Module):
    """Parallel morphology filters followed by deterministic 16x reduction."""

    def __init__(self, in_channels: int = 1, hidden: int = 128, dropout: float = 0.10):
        super().__init__()
        branch = hidden // 4
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, branch, kernel, stride=4, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(branch),
                    nn.GELU(),
                )
                for kernel in (5, 9, 17, 33)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(4 * branch, hidden, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            ResidualTemporalBlock(hidden, hidden, dilation=1, dropout=dropout),
            ResidualTemporalBlock(hidden, hidden, dilation=2, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class AttentiveStatisticsPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(channels, channels // 2), nn.Tanh(), nn.Linear(channels // 2, 1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=-1)
        mean = torch.sum(sequence * weights[..., None], dim=1)
        variance = torch.sum((sequence - mean[:, None]) ** 2 * weights[..., None], dim=1).clamp_min(1e-6)
        return torch.cat([mean, torch.sqrt(variance)], dim=-1)


class CNNBiLSTM(nn.Module):
    """Multiscale residual CNN-BiLSTM comparator under the common protocol."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        branch_hidden = 96
        self.ecg_stem = MultiScaleStem(1, branch_hidden)
        self.ppg_stem = MultiScaleStem(1, branch_hidden)
        self.fusion = nn.Sequential(
            nn.Conv1d(2 * branch_hidden, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            ResidualTemporalBlock(hidden, hidden, dilation=4),
        )
        self.rnn = nn.LSTM(hidden, 96, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
        self.self_attention = nn.MultiheadAttention(192, 6, dropout=0.10, batch_first=True)
        self.pool = AttentiveStatisticsPool(192)
        self.head_hidden = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 192), nn.GELU(), nn.Dropout(0.10), nn.Linear(192, 128), nn.GELU())
        self.head_out = nn.Linear(128, 2)

    def regression_features(self, ecg, ppg, sqi=None):
        z = self.fusion(torch.cat([self.ecg_stem(ecg), self.ppg_stem(ppg)], dim=1)).transpose(1, 2)
        z, _ = self.rnn(z)
        attended, _ = self.self_attention(z, z, z, need_weights=False)
        return self.head_hidden(self.pool(z + attended))

    def forward(self, ecg, ppg, sqi=None):
        return self.head_out(self.regression_features(ecg, ppg, sqi))


class BPNet(nn.Module):
    """Deep dual-channel residual/dilated BP-Net common-protocol adaptation."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.ecg_entry = nn.Conv1d(1, 48, 11, stride=4, padding=5, bias=False)
        self.ppg_entry = nn.Conv1d(1, 48, 11, stride=4, padding=5, bias=False)
        self.ecg_tower = self._tower()
        self.ppg_tower = self._tower()
        self.cross_gate = nn.Sequential(nn.Conv1d(256, 256, 1), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Conv1d(256, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            *[ResidualTemporalBlock(hidden, hidden, dilation=d) for d in (1, 2, 4, 8, 4, 2)],
        )
        self.pool = AttentiveStatisticsPool(hidden)
        self.head_hidden = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, 192), nn.GELU(), nn.Dropout(0.10), nn.Linear(192, 128), nn.GELU())
        self.head_out = nn.Linear(128, 2)

    @staticmethod
    def _tower() -> nn.Sequential:
        return nn.Sequential(
            ResidualTemporalBlock(48, 64, stride=2),
            ResidualTemporalBlock(64, 64, dilation=2),
            ResidualTemporalBlock(64, 96, stride=2),
            ResidualTemporalBlock(96, 96, dilation=4),
            ResidualTemporalBlock(96, 128, dilation=8),
            ResidualTemporalBlock(128, 128, dilation=4),
        )

    def regression_features(self, ecg, ppg, sqi=None):
        e = self.ecg_tower(self.ecg_entry(ecg))
        p = self.ppg_tower(self.ppg_entry(ppg))
        pair = torch.cat([e, p], dim=1)
        gate = self.cross_gate(pair)
        mixed = torch.cat([e + gate[:, :128] * p, p + gate[:, 128:] * e], dim=1)
        z = self.fusion(mixed).transpose(1, 2)
        return self.head_hidden(self.pool(z))

    def forward(self, ecg, ppg, sqi=None):
        return self.head_out(self.regression_features(ecg, ppg, sqi))


class AttentionGatedGRULayer(nn.Module):
    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, batch_first=True, bidirectional=True)
        self.gate = nn.Sequential(nn.Linear(2 * hidden + input_size, 2 * hidden), nn.Sigmoid())
        self.residual = nn.Linear(input_size, 2 * hidden) if input_size != 2 * hidden else nn.Identity()
        self.norm = nn.LayerNorm(2 * hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.gru(x)
        gated = self.gate(torch.cat([x, recurrent], dim=-1)) * recurrent
        return self.norm(self.residual(x) + gated)


class TESAGRU(nn.Module):
    """Temporal encoder, self/cross-attention, and stacked attention-gated GRU."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        branch_hidden = 96
        self.ecg = MultiScaleStem(1, branch_hidden)
        self.ppg = MultiScaleStem(1, branch_hidden)
        self.ecg_temporal = nn.Sequential(*[ResidualTemporalBlock(branch_hidden, branch_hidden, dilation=d) for d in (1, 2, 4)])
        self.ppg_temporal = nn.Sequential(*[ResidualTemporalBlock(branch_hidden, branch_hidden, dilation=d) for d in (1, 2, 4)])
        self.ecg_to_ppg = nn.MultiheadAttention(branch_hidden, 4, dropout=0.10, batch_first=True)
        self.ppg_to_ecg = nn.MultiheadAttention(branch_hidden, 4, dropout=0.10, batch_first=True)
        layer = nn.TransformerEncoderLayer(2 * branch_hidden, 6, 4 * branch_hidden, dropout=0.10, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.gru_1 = AttentionGatedGRULayer(2 * branch_hidden, 96)
        self.gru_2 = AttentionGatedGRULayer(192, 96)
        self.pool = AttentiveStatisticsPool(192)
        self.head_hidden = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 192), nn.GELU(), nn.Dropout(0.10), nn.Linear(192, 128), nn.GELU())
        self.head_out = nn.Linear(128, 2)

    def regression_features(self, ecg, ppg, sqi=None):
        e = self.ecg_temporal(self.ecg(ecg)).transpose(1, 2)
        p = self.ppg_temporal(self.ppg(ppg)).transpose(1, 2)
        ep, _ = self.ecg_to_ppg(e, p, p, need_weights=False)
        pe, _ = self.ppg_to_ecg(p, e, e, need_weights=False)
        z = self.encoder(torch.cat([e + ep, p + pe], dim=-1))
        z = self.gru_2(self.gru_1(z))
        return self.head_hidden(self.pool(z))

    def forward(self, ecg, ppg, sqi=None):
        return self.head_out(self.regression_features(ecg, ppg, sqi))


class DynamicFeatureStream(nn.Module):
    """Learned waveform/derivative stream used by the MuFu-style comparator."""

    def __init__(self, hidden: int):
        super().__init__()
        self.stem = MultiScaleStem(2, hidden)
        self.context = nn.Sequential(*[ResidualTemporalBlock(hidden, hidden, dilation=d) for d in (1, 2, 4, 8)])

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        derivative = F.pad(signal[..., 1:] - signal[..., :-1], (1, 0))
        return self.context(self.stem(torch.cat([signal, derivative], dim=1))).transpose(1, 2)


class MuFuBPNet(nn.Module):
    """Morphology/dynamics dual-stream probabilistic fusion implementation."""

    def __init__(self, hidden: int = 128, latent: int = 96):
        super().__init__()
        self.ecg_morphology = MultiScaleStem(1, hidden)
        self.ppg_morphology = MultiScaleStem(1, hidden)
        self.ecg_dynamics = DynamicFeatureStream(hidden)
        self.ppg_dynamics = DynamicFeatureStream(hidden)
        self.ecg_cross = nn.MultiheadAttention(2 * hidden, 8, dropout=0.10, batch_first=True)
        self.ppg_cross = nn.MultiheadAttention(2 * hidden, 8, dropout=0.10, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(4 * hidden, 8, 4 * hidden, dropout=0.10, batch_first=True, norm_first=True),
            2,
            enable_nested_tensor=False,
        )
        self.pool = AttentiveStatisticsPool(4 * hidden)
        pooled_dim = 8 * hidden
        self.mu = nn.Linear(pooled_dim, latent)
        self.logvar = nn.Linear(pooled_dim, latent)
        self.deterministic_projection = nn.Sequential(nn.LayerNorm(pooled_dim), nn.Linear(pooled_dim, 256), nn.GELU(), nn.Dropout(0.10))
        self.fusion_hidden = nn.Sequential(nn.Linear(256 + latent, 256), nn.GELU(), nn.Dropout(0.10), nn.Linear(256, 192), nn.GELU())
        self.fusion_out = nn.Linear(192, 2)

    def _latent(self, ecg, ppg):
        e = torch.cat([self.ecg_morphology(ecg).transpose(1, 2), self.ecg_dynamics(ecg)], dim=-1)
        p = torch.cat([self.ppg_morphology(ppg).transpose(1, 2), self.ppg_dynamics(ppg)], dim=-1)
        ep, _ = self.ecg_cross(e, p, p, need_weights=False)
        pe, _ = self.ppg_cross(p, e, e, need_weights=False)
        fused = self.fusion_encoder(torch.cat([e + ep, p + pe], dim=-1))
        pooled = self.pool(fused)
        mu = self.mu(pooled)
        logvar = self.logvar(pooled).clamp(-8, 5)
        latent = mu if not self.training else mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        deterministic = self.deterministic_projection(pooled)
        return deterministic, mu, logvar, latent

    def regression_features(self, ecg, ppg, sqi=None):
        deterministic, _, _, latent = self._latent(ecg, ppg)
        return self.fusion_hidden(torch.cat([deterministic, latent], dim=-1))

    def forward_with_aux(self, ecg, ppg, target=None):
        deterministic, mu, logvar, latent = self._latent(ecg, ppg)
        features = self.fusion_hidden(torch.cat([deterministic, latent], dim=-1))
        return self.fusion_out(features), {"mu": mu, "logvar": logvar}

    def forward(self, ecg, ppg, sqi=None):
        return self.fusion_out(self.regression_features(ecg, ppg, sqi))


class PATRidge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_ = None

    def fit(self, features, targets):
        x = np.asarray(features, float)
        y = np.asarray(targets, float)
        x = np.column_stack([x, np.ones(len(x))])
        penalty = self.alpha * np.eye(x.shape[1])
        penalty[-1, -1] = 0
        self.coef_ = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        return self

    def predict(self, features):
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        x = np.column_stack([np.asarray(features, float), np.ones(len(features))])
        return x @ self.coef_


def pat_ridge_features(ecg: np.ndarray, ppg: np.ndarray, sample_rate_hz: float = 250.0) -> np.ndarray:
    """PAT, heart rate and missing-PAT indicator for the ridge comparator."""
    ecg = np.asarray(ecg, dtype=float)
    ppg = np.asarray(ppg, dtype=float)
    if ecg.ndim == 1:
        ecg = ecg[None]
    if ppg.ndim == 1:
        ppg = ppg[None]
    if ecg.shape != ppg.shape:
        raise ValueError("ECG and PPG must have identical batch shapes")
    rows = []
    minimum_lag = int(round(0.060 * sample_rate_hz))
    maximum_lag = int(round(0.650 * sample_rate_hz))
    for e, p in zip(ecg, ppg, strict=True):
        energy = np.gradient(e) ** 2
        peaks, _ = signal.find_peaks(energy, distance=int(0.28 * sample_rate_hz), prominence=max(np.std(energy) * 0.20, 1e-9))
        heart_rate = 60.0 * sample_rate_hz / np.median(np.diff(peaks)) if len(peaks) >= 3 else 0.0
        upstroke = np.maximum(np.gradient(p), 0.0)
        delays = []
        for peak in peaks:
            left = peak + minimum_lag
            right = min(peak + maximum_lag + 1, len(p))
            if right > left:
                delays.append(1000.0 * (left + int(np.argmax(upstroke[left:right])) - peak) / sample_rate_hz)
        missing = len(delays) < 2
        pat_ms = float(np.median(delays)) if not missing else 0.0
        rows.append([pat_ms, float(heart_rate), float(missing)])
    return np.asarray(rows, dtype=np.float32)


def build_neural_baseline(name: str) -> nn.Module:
    registry = {"cnn_bilstm": CNNBiLSTM, "bp_net": BPNet, "te_sagru": TESAGRU, "mufubp_net": MuFuBPNet}
    if name not in registry:
        raise KeyError(f"Unknown baseline: {name}")
    return registry[name]()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def engineered_features(ecg: np.ndarray, ppg: np.ndarray, sample_rate_hz: float = 250.0) -> np.ndarray:
    """Deterministic PAT-style features used by the classical comparator.

    Features are computed independently per window and never use ABP.  The
    implementation deliberately exposes the complete feature vector so that
    the classical baseline can be reproduced without a hidden feature cache.
    """
    ecg = np.asarray(ecg, dtype=float)
    ppg = np.asarray(ppg, dtype=float)
    if ecg.ndim == 1:
        ecg = ecg[None, :]
    if ppg.ndim == 1:
        ppg = ppg[None, :]
    if ecg.shape != ppg.shape:
        raise ValueError("ECG and PPG must have identical batch shapes")
    rows = []
    for e, p in zip(ecg, ppg):
        e0, p0 = e - np.median(e), p - np.median(p)
        e_scale = np.percentile(e0, 95) - np.percentile(e0, 5) + 1e-6
        p_scale = np.percentile(p0, 95) - np.percentile(p0, 5) + 1e-6
        de = np.diff(e0, prepend=e0[0])
        dp = np.diff(p0, prepend=p0[0])
        ecg_peak = int(np.argmax(np.abs(de)))
        ppg_peak = int(np.argmax(np.abs(dp)))
        pat_ms = (ppg_peak - ecg_peak) * 1000.0 / float(sample_rate_hz)
        feats = [
            float(np.mean(e0)), float(np.std(e0)), float(np.mean(np.abs(de))),
            float(np.percentile(e0, 95) / e_scale), float(np.percentile(e0, 5) / e_scale),
            float(np.mean(p0)), float(np.std(p0)), float(np.mean(np.abs(dp))),
            float(np.percentile(p0, 95) / p_scale), float(np.percentile(p0, 5) / p_scale),
            float(np.clip(pat_ms, -1000.0, 1000.0)), float(np.corrcoef(e0, p0)[0, 1]) if np.std(e0) and np.std(p0) else 0.0,
        ]
        rows.append(feats)
    return np.asarray(rows, dtype=np.float32)


class EngineeredRandomForest:
    """Explicit classical comparator with fold-local feature fitting."""
    def __init__(self, n_estimators: int = 500, random_state: int = 42, min_samples_leaf: int = 2):
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.min_samples_leaf = int(min_samples_leaf)
        self.models = []

    def fit(self, ecg, ppg, targets):
        from sklearn.ensemble import RandomForestRegressor
        x = engineered_features(ecg, ppg)
        y = np.asarray(targets, dtype=float)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("targets must have shape [n,2]")
        self.models = [
            RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state + j,
                                  min_samples_leaf=self.min_samples_leaf, n_jobs=1)
            .fit(x, y[:, j]) for j in range(2)
        ]
        return self

    def predict(self, ecg, ppg):
        if len(self.models) != 2:
            raise RuntimeError("fit must be called before predict")
        x = engineered_features(ecg, ppg)
        return np.column_stack([model.predict(x) for model in self.models])
