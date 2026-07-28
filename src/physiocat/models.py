from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PhysioCATConfig:
    samples_per_window: int = 2000
    n_tokens: int = 125
    hidden: int = 128
    heads: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.10
    token_ms: int = 64
    lower_ms: int = 120
    upper_ms: int = 450
    patch_samples: int = 16
    patch_mlp_multiplier: int = 2
    temporal_layers: int = 2
    mask_kind: str = "delay_asymmetric"
    mask_seed: int = 82000
    use_sqi_fusion: bool = True


def active_query_rows(n_tokens: int = 125, start: int = 0, stop: int | None = None, *, device=None) -> torch.Tensor:
    """Compatibility helper: all sequence tokens remain eligible for pooling."""
    stop = n_tokens if stop is None else min(int(stop), int(n_tokens))
    rows = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    rows[max(0, int(start)):stop] = True
    return rows


def _offset_mask(offsets: tuple[int, ...], n_tokens: int, *, device=None) -> torch.Tensor:
    query = torch.arange(n_tokens, device=device)[:, None]
    key = torch.arange(n_tokens, device=device)[None, :]
    allowed = torch.zeros((n_tokens, n_tokens), dtype=torch.bool, device=device)
    for offset in offsets:
        allowed |= key == query + int(offset)
    return allowed


def _degree_preserving_random_rewire(reference: torch.Tensor, seed: int, *, device=None) -> torch.Tensor:
    """Randomize a bipartite mask without changing either endpoint degree.

    A sequence of valid double-edge swaps preserves the complete row-degree
    and column-degree vectors, the total number of edges, and the active-token
    support on both modalities.  The operation is deterministic for a topology
    seed and starts from the exact reference graph, making it a stricter
    sparsity/topology control than independent per-query sampling.
    """
    result = reference.detach().to(device="cpu", dtype=torch.bool).clone()
    edges = torch.nonzero(result, as_tuple=False)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    target_swaps = max(1, int(edges.shape[0]) * 20)
    accepted = 0
    attempts = 0
    max_attempts = target_swaps * 40
    while accepted < target_swaps and attempts < max_attempts:
        attempts += 1
        chosen = torch.randint(edges.shape[0], (2,), generator=generator)
        first = int(chosen[0])
        second = int(chosen[1])
        if first == second:
            continue
        row_a, col_a = map(int, edges[first].tolist())
        row_b, col_b = map(int, edges[second].tolist())
        if row_a == row_b or col_a == col_b:
            continue
        if bool(result[row_a, col_b]) or bool(result[row_b, col_a]):
            continue
        result[row_a, col_a] = False
        result[row_b, col_b] = False
        result[row_a, col_b] = True
        result[row_b, col_a] = True
        edges[first, 1] = col_b
        edges[second, 1] = col_a
        accepted += 1
    if accepted < target_swaps:
        raise RuntimeError(
            f"Degree-preserving random rewiring stalled after {accepted}/{target_swaps} swaps"
        )
    if not torch.equal(result.sum(0), reference.cpu().sum(0)):
        raise RuntimeError("Random rewiring changed the key-side degree vector")
    if not torch.equal(result.sum(1), reference.cpu().sum(1)):
        raise RuntimeError("Random rewiring changed the query-side degree vector")
    return result.to(device=device)


def delay_mask(
    n_tokens: int = 125,
    token_ms: int = 64,
    lower_ms: int = 120,
    upper_ms: int = 450,
    *,
    reverse: bool = False,
    active_start: int = 0,
    active_stop: int | None = None,
    device=None,
) -> torch.Tensor:
    """Support-conservative physiologic delay mask.

    Forward rows are ECG queries and use later PPG keys. Reverse rows are PPG
    queries and use earlier ECG keys. Both branches therefore encode the same
    ECG-leading-PPG relation without treating the modalities as exchangeable.

    A token covers one non-overlapping 16-sample patch.  Offsets are retained
    only when every analysis-grid sample pair across the two patches lies inside the
    declared 120--450 ms interval.  At 250 Hz this yields token offsets 3--6,
    whose realized analysis-grid patch support is 132--444 ms.  This
    conservative rule avoids presenting token-centre spacing as a stricter
    event-level guarantee.
    """
    samples_per_token = 16
    sample_ms = float(token_ms) / samples_per_token
    within_patch_span_ms = float(token_ms) - sample_ms
    lower_offset = int(math.ceil((float(lower_ms) + within_patch_span_ms) / float(token_ms)))
    upper_offset = int(math.floor((float(upper_ms) - within_patch_span_ms) / float(token_ms)))
    if lower_offset > upper_offset:
        raise ValueError("Delay interval is too narrow for support-conservative patch interactions")
    offsets = tuple(range(lower_offset, upper_offset + 1))
    if reverse:
        offsets = tuple(-value for value in offsets)
    allowed = _offset_mask(offsets, n_tokens, device=device)
    rows = active_query_rows(n_tokens, active_start, active_stop, device=device)
    return allowed & rows[:, None]


def mechanism_mask(
    kind: str,
    n_tokens: int = 125,
    token_ms: int = 64,
    lower_ms: int = 120,
    upper_ms: int = 450,
    *,
    reverse: bool = False,
    seed: int = 82000,
    active_start: int = 0,
    active_stop: int | None = None,
    device=None,
) -> torch.Tensor:
    """Implementation-level mask for every reported mechanism control.

    The random control preserves both endpoint degree vectors of the default
    bipartite graph. The PPG-leading mirror reverses the sign of every default
    lag while preserving the exact parameterization and total edge count. The
    zero-centred local control contains the four strictly nonzero offsets
    {-2,-1,+1,+2}, so it cannot acquire a direction preference through a
    same-time tie. Equal-width location controls use a common ECG-anchor
    support, which isolates band position from edge capacity and sequence-edge
    truncation.
    """
    if kind == "delay_asymmetric":
        return delay_mask(n_tokens, token_ms, lower_ms, upper_ms, reverse=reverse,
                          active_start=active_start, active_stop=active_stop, device=device)
    if kind == "no_delay":
        reference = delay_mask(n_tokens, token_ms, lower_ms, upper_ms, reverse=reverse, device=device)
        rows = reference.sum(1) > 0
        return rows[:, None].expand(n_tokens, n_tokens).clone()
    if kind in {
        "shifted",
        "offsets_2_5_common",
        "offsets_3_6_common",
        "offsets_4_7_common",
        "offsets_9_12_common",
    }:
        lower, upper = {
            "shifted": (9, 12),
            "offsets_2_5_common": (2, 5),
            "offsets_3_6_common": (3, 6),
            "offsets_4_7_common": (4, 7),
            "offsets_9_12_common": (9, 12),
        }[kind]
        offsets = tuple(range(lower, upper + 1))
        if reverse:
            offsets = tuple(-value for value in offsets)
        allowed = _offset_mask(offsets, n_tokens, device=device)
        # Use the same ECG-anchor support (tokens 0--112) for every four-offset
        # band, including the remote 9--12 control. In the reciprocal branch
        # the ECG-anchor support is on the key axis, making that branch the
        # exact transpose of the forward graph. Every condition therefore has
        # exactly 113 x 4 = 452 edges and the same endpoint-degree distribution
        # up to a translation of the PPG-key support.
        index = torch.arange(n_tokens, device=device)
        common_anchor = index <= 112
        return allowed & (common_anchor[None, :] if reverse else common_anchor[:, None])
    if kind == "local":
        # A truly direction-agnostic local graph. Excluding offset zero avoids
        # the deterministic earlier-key tie that arises when an even number of
        # nearest keys is selected around a same-time edge. The graph is
        # symmetric, so the reciprocal branch is exactly its transpose.
        return _offset_mask((-2, -1, 1, 2), n_tokens, device=device)
    if kind == "random":
        forward_reference = delay_mask(
            n_tokens, token_ms, lower_ms, upper_ms, reverse=False, device=device
        )
        forward_random = _degree_preserving_random_rewire(
            forward_reference, seed, device=device
        )
        return forward_random.transpose(0, 1).clone() if reverse else forward_random
    if kind == "mirror":
        return delay_mask(
            n_tokens,
            token_ms,
            lower_ms,
            upper_ms,
            reverse=not reverse,
            active_start=active_start,
            active_stop=active_stop,
            device=device,
        )
    if kind == "unidirectional":
        if reverse:
            return torch.zeros((n_tokens, n_tokens), dtype=torch.bool, device=device)
        return delay_mask(n_tokens, token_ms, lower_ms, upper_ms, reverse=False,
                          active_start=active_start, active_stop=active_stop, device=device)
    raise KeyError(f"Unknown mechanism mask: {kind}")


def mask_row_audit(mask: torch.Tensor) -> dict[str, int | list[int]]:
    counts = mask.to(torch.int64).sum(dim=1).cpu()
    return {
        "total_edges": int(counts.sum()),
        "active_rows": int((counts > 0).sum()),
        "empty_rows": int((counts == 0).sum()),
        "unique_row_counts": sorted(map(int, torch.unique(counts).tolist())),
    }


def _token_sqi(sqi: torch.Tensor, n_tokens: int) -> torch.Tensor:
    if sqi.ndim == 2 and sqi.shape[1] == 2:
        return sqi[:, :, None].expand(-1, -1, n_tokens)
    if sqi.ndim != 3 or sqi.shape[1] != 2:
        raise ValueError("sqi must have shape [batch,2] or [batch,2,tokens]")
    if sqi.shape[-1] != n_tokens:
        sqi = F.interpolate(sqi, size=n_tokens, mode="linear", align_corners=False)
    return sqi.clamp(0, 1)


class LocalPatchEncoder(nn.Module):
    """Encode one non-overlapping 64-ms waveform patch per token.

    A stride-equal-to-kernel convolution is a learned local filter bank over
    exactly 16 input samples.  The residual MLP acts independently at every
    token, so no information can cross a patch boundary before the explicit
    delay-masked ECG--PPG exchange.
    """
    receptive_field_samples = 16

    def __init__(
        self,
        hidden: int = 128,
        mlp_multiplier: int = 2,
        dropout: float = 0.10,
        patch_samples: int = 16,
    ):
        super().__init__()
        self.patch_samples = int(patch_samples)
        expanded = int(hidden * mlp_multiplier)
        self.patch_projection = nn.Conv1d(
            1,
            hidden,
            kernel_size=self.patch_samples,
            stride=self.patch_samples,
            padding=0,
            bias=False,
        )
        self.input_norm = nn.LayerNorm(hidden)
        self.token_mlp = nn.Sequential(
            nn.Linear(hidden, expanded),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded, hidden),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 1:
            raise ValueError("signal must have shape [batch,1,samples]")
        if signal.shape[-1] % self.patch_samples:
            raise ValueError("Signal length must be divisible by patch_samples")
        tokens = self.patch_projection(signal).transpose(1, 2)
        normalized = self.input_norm(tokens)
        return self.output_norm(tokens + self.token_mlp(normalized))


class PostFusionTemporalEncoder(nn.Module):
    """Model whole-window morphology after strictly band-limited exchange."""

    def __init__(self, config: PhysioCATConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden,
                    nhead=config.heads,
                    dim_feedforward=config.hidden * config.ffn_multiplier,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.temporal_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden)

    def forward(self, tokens: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        state = tokens
        for layer in self.layers:
            state = layer(state, src_key_padding_mask=src_key_padding_mask)
        return self.final_norm(state)


class SafeCrossAttentionBranch(nn.Module):
    """Masked multi-head attention with exactly zero output for inactive rows."""

    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def attention_weights(self, query: torch.Tensor, key_value: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        batch, n_query, _ = query.shape
        n_key = key_value.shape[1]
        q = self.q(query).view(batch, n_query, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(key_value).view(batch, n_key, self.heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = allowed.to(device=scores.device, dtype=torch.bool)
        valid_rows = allowed.any(dim=-1)
        scores = scores.masked_fill(~allowed[None, None], -torch.inf)
        scores = torch.where(valid_rows[None, None, :, None], scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(valid_rows[None, None, :, None], weights, torch.zeros_like(weights))
        return weights

    def projected_values(self, key_value: torch.Tensor) -> torch.Tensor:
        batch, n_key, _ = key_value.shape
        return self.v(key_value).view(batch, n_key, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n_query, _ = query.shape
        weights = self.attention_weights(query, key_value, allowed)
        v = self.projected_values(key_value)
        output = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, n_query, self.hidden)
        output = self.out(output)
        valid_rows = allowed.to(device=output.device, dtype=torch.bool).any(dim=-1)
        output = torch.where(valid_rows[None, :, None], output, torch.zeros_like(output))
        return output, weights


class PhysioCAT(nn.Module):
    def __init__(
        self,
        config: PhysioCATConfig | None = None,
        *,
        mask_kind: str | None = None,
        use_sqi_fusion: bool | None = None,
        fixed_uniform_affinity: bool = False,
    ):
        super().__init__()
        self.config = config or PhysioCATConfig()
        self.mask_kind = mask_kind or self.config.mask_kind
        self.use_sqi_fusion = self.config.use_sqi_fusion if use_sqi_fusion is None else bool(use_sqi_fusion)
        self.fixed_uniform_affinity = bool(fixed_uniform_affinity)
        c = self.config
        self.ecg_stem = LocalPatchEncoder(c.hidden, c.patch_mlp_multiplier, c.dropout, c.patch_samples)
        self.ppg_stem = LocalPatchEncoder(c.hidden, c.patch_mlp_multiplier, c.dropout, c.patch_samples)
        self.position_embedding = nn.Parameter(torch.zeros(1, c.n_tokens, c.hidden))
        self.modality_embedding = nn.Parameter(torch.zeros(2, 1, c.hidden))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        self.ecg_queries_ppg = SafeCrossAttentionBranch(c.hidden, c.heads, c.dropout)
        self.ppg_queries_ecg = SafeCrossAttentionBranch(c.hidden, c.heads, c.dropout)
        self.edge_pair_projection = nn.Linear(2 * c.hidden, c.hidden)
        self.edge_pair_norm = nn.LayerNorm(c.hidden)
        self.edge_dropout = nn.Dropout(c.dropout)
        self.fusion_norm = nn.LayerNorm(c.hidden)
        self.temporal_encoder = PostFusionTemporalEncoder(c)
        self.attentive_pool = nn.Linear(c.hidden, 1)
        self.regressor_hidden = nn.Sequential(nn.LayerNorm(3 * c.hidden), nn.Linear(3 * c.hidden, 128), nn.GELU(), nn.Dropout(c.dropout))
        self.regressor_out = nn.Linear(128, 2)

    def active_tokens(self, n_tokens: int, device) -> torch.Tensor:
        return active_query_rows(n_tokens, device=device)

    def _allowed(self, n_tokens: int, reverse: bool, device) -> torch.Tensor:
        return mechanism_mask(
            self.mask_kind, n_tokens, self.config.token_ms, self.config.lower_ms, self.config.upper_ms,
            reverse=reverse, seed=self.config.mask_seed, device=device,
        )

    def encode(self, ecg: torch.Tensor, ppg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.ecg_stem(ecg)
        p = self.ppg_stem(ppg)
        if e.shape[1] != self.config.n_tokens or p.shape[1] != self.config.n_tokens:
            raise ValueError(f"Expected {self.config.n_tokens} tokens; got ECG={e.shape[1]}, PPG={p.shape[1]}")
        e = e + self.position_embedding + self.modality_embedding[0]
        p = p + self.position_embedding + self.modality_embedding[1]
        return e, p

    def edge_aligned_pair_message(self, ecg: torch.Tensor, ppg: torch.Tensor, sqi: torch.Tensor):
        """Fuse reciprocal attention evidence on the same ECG-leading edges.

        For every admissible pair (ECG i, PPG j), the ECG-query and PPG-query
        branches score that exact edge in opposite query/key assignments. Their
        geometric-mean affinity is normalized over the admissible future PPG
        partners of ECG anchor i.  The edge SQI is the geometric mean of the
        two tokens' scale-invariant quality scores, so neither a clean query nor
        a clean key/value can conceal corruption in its partner.
        """
        e, p = self.encode(ecg, ppg)
        pair_allowed = self._allowed(e.shape[1], False, e.device)
        reverse_allowed = pair_allowed.transpose(0, 1)
        if self.fixed_uniform_affinity:
            forward = pair_allowed.to(dtype=e.dtype)
            forward = forward / forward.sum(dim=-1, keepdim=True).clamp_min(1.0)
            reverse = reverse_allowed.to(dtype=e.dtype)
            reverse = reverse / reverse.sum(dim=-1, keepdim=True).clamp_min(1.0)
            weights_ecg_query = forward[None, None].expand(e.shape[0], self.config.heads, -1, -1)
            weights_ppg_query = reverse[None, None].expand(e.shape[0], self.config.heads, -1, -1)
            pair_affinity = weights_ecg_query
        else:
            weights_ecg_query = self.ecg_queries_ppg.attention_weights(e, p, pair_allowed)
        if self.fixed_uniform_affinity:
            pass
        elif self.mask_kind == "unidirectional":
            weights_ppg_query = torch.zeros_like(weights_ecg_query.transpose(-2, -1))
            pair_affinity = weights_ecg_query
        else:
            weights_ppg_query = self.ppg_queries_ecg.attention_weights(p, e, reverse_allowed)
            reciprocal_product = (
                torch.clamp(weights_ecg_query, min=0.0)
                * torch.clamp(weights_ppg_query.transpose(-2, -1), min=0.0)
            )
            # The mask creates exact zeros. Clamp before sqrt so the backward
            # derivative remains finite; the admissible-edge mask below still
            # restores exact zero support outside the declared pair relation.
            pair_affinity = torch.sqrt(reciprocal_product.clamp_min(1e-12))

        q = _token_sqi(sqi, e.shape[1])
        edge_reliability = torch.sqrt(
            torch.clamp(q[:, 0, :, None] * q[:, 1, None, :], min=0.0, max=1.0)
        )
        pair_affinity = pair_affinity * pair_allowed[None, None]
        base_denominator = pair_affinity.sum(dim=-1, keepdim=True)
        base_weights = torch.where(
            base_denominator > 0,
            pair_affinity / base_denominator.clamp_min(torch.finfo(pair_affinity.dtype).eps),
            torch.zeros_like(pair_affinity),
        )
        if self.use_sqi_fusion:
            pair_affinity = pair_affinity * edge_reliability[:, None]
        denominator = pair_affinity.sum(dim=-1, keepdim=True)
        pair_weights = torch.where(
            denominator > 0,
            pair_affinity / denominator.clamp_min(torch.finfo(pair_affinity.dtype).eps),
            torch.zeros_like(pair_affinity),
        )

        p_values = self.ecg_queries_ppg.projected_values(p)
        e_values = self.ppg_queries_ecg.projected_values(e)
        p_message = (self.edge_dropout(pair_weights) @ p_values).transpose(1, 2).reshape(e.shape[0], e.shape[1], -1)
        e_message = e_values.transpose(1, 2).reshape(e.shape[0], e.shape[1], -1)
        p_message = self.ecg_queries_ppg.out(p_message)
        e_message = self.ppg_queries_ecg.out(e_message)
        pair_message = self.edge_pair_norm(self.edge_pair_projection(torch.cat([e_message, p_message], dim=-1)))
        active = pair_allowed.any(dim=-1)
        pair_message = torch.where(active[None, :, None], pair_message, torch.zeros_like(pair_message))
        if self.use_sqi_fusion:
            anchor_reliability = (base_weights * edge_reliability[:, None]).sum(dim=-1).mean(dim=1)
        else:
            anchor_reliability = active[None].expand(e.shape[0], -1).to(dtype=pair_message.dtype)
        attention = {
            "ecg_query_ppg_key_value": weights_ecg_query,
            "ppg_query_ecg_key_value": weights_ppg_query,
            "edge_aligned_pair": pair_weights,
            "edge_pair_reliability": edge_reliability,
            "edge_anchor_reliability": anchor_reliability,
            "active_rows": active,
        }
        return pair_message, attention

    def pre_temporal_fusion(self, ecg: torch.Tensor, ppg: torch.Tensor, sqi: torch.Tensor):
        pair_message, attention = self.edge_aligned_pair_message(ecg, ppg, sqi)
        pre_temporal = self.fusion_norm(pair_message) * attention["edge_anchor_reliability"][..., None]
        return pre_temporal, attention

    def fusion_tokens(self, ecg: torch.Tensor, ppg: torch.Tensor, sqi: torch.Tensor, *, return_attention: bool = False):
        pre_temporal, attention = self.pre_temporal_fusion(ecg, ppg, sqi)
        active = attention["active_rows"]
        padding = (~active)[None].expand(pre_temporal.shape[0], -1)
        fused = self.temporal_encoder(pre_temporal, src_key_padding_mask=padding)
        fused = torch.where(active[None, :, None], fused, torch.zeros_like(fused))
        if return_attention:
            return fused, attention
        return fused

    def _active_pool(self, fused: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        if active is None:
            active = torch.ones(fused.shape[1], dtype=torch.bool, device=fused.device)
        scores = self.attentive_pool(fused).squeeze(-1).masked_fill(~active[None], -torch.inf)
        pool_weights = torch.softmax(scores, dim=-1)
        attentive = torch.sum(fused * pool_weights[..., None], dim=1)
        active_float = active[None, :, None].to(dtype=fused.dtype)
        average = (fused * active_float).sum(dim=1) / active_float.sum(dim=1).clamp_min(1.0)
        maximum = fused.masked_fill(~active[None, :, None], -torch.inf).max(dim=1).values
        return torch.cat([attentive, average, maximum], dim=-1)

    def pooled_features(self, ecg: torch.Tensor, ppg: torch.Tensor, sqi: torch.Tensor) -> torch.Tensor:
        fused, attention = self.fusion_tokens(ecg, ppg, sqi, return_attention=True)
        return self._active_pool(fused, attention.get("active_rows"))

    def regression_features(self, ecg: torch.Tensor, ppg: torch.Tensor, sqi: torch.Tensor) -> torch.Tensor:
        return self.regressor_hidden(self.pooled_features(ecg, ppg, sqi))

    def contrastive_features(self, ecg: torch.Tensor, ppg: torch.Tensor, sqi: torch.Tensor) -> dict[str, torch.Tensor]:
        e, p = self.encode(ecg, ppg)
        return {"ecg": e.mean(dim=1), "ppg": p.mean(dim=1)}

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor, sqi: torch.Tensor, *, return_attention: bool = False):
        if return_attention:
            fused, attention = self.fusion_tokens(ecg, ppg, sqi, return_attention=True)
            return self.regressor_out(self.regressor_hidden(self._active_pool(fused, attention.get("active_rows")))), attention
        return self.regressor_out(self.regression_features(ecg, ppg, sqi))


class MatchedNoDelayPhysioCAT(PhysioCAT):
    def __init__(self, config: PhysioCATConfig | None = None):
        super().__init__(config=config, mask_kind="no_delay")


class UniformDelayBandPhysioCAT(PhysioCAT):
    """Non-learned within-band affinity with the same value/SQI pathway."""

    def __init__(self, config: PhysioCATConfig | None = None):
        super().__init__(
            config=config,
            mask_kind="delay_asymmetric",
            fixed_uniform_affinity=True,
        )
        # Query/key projections are unnecessary for a fixed affinity control;
        # value and output projections, edge SQI, temporal encoder, and head are
        # retained so the comparison isolates morphology-dependent edge scores.
        del self.ecg_queries_ppg.q, self.ecg_queries_ppg.k
        del self.ppg_queries_ecg.q, self.ppg_queries_ecg.k


class InputAblationPhysioCAT(PhysioCAT):
    """True single-modality control with no hidden access to the other stream."""

    def __init__(self, modality: str, config: PhysioCATConfig | None = None):
        super().__init__(config=config, mask_kind="delay_asymmetric")
        if modality not in {"ecg", "ppg"}:
            raise ValueError(modality)
        self.modality = modality
        # Remove every parameter belonging exclusively to the unavailable
        # modality or to cross-modal fusion.  A zero waveform would still
        # carry positional/modality embeddings and is therefore not a valid
        # single-input ablation.
        if modality == "ecg":
            del self.ppg_stem
        else:
            del self.ecg_stem
        del self.ecg_queries_ppg, self.ppg_queries_ecg
        del self.edge_pair_projection, self.edge_pair_norm, self.edge_dropout

    def _single_tokens(self, ecg: torch.Tensor, ppg: torch.Tensor) -> torch.Tensor:
        if self.modality == "ecg":
            tokens = self.ecg_stem(ecg)
            embedding = self.modality_embedding[0]
            encoded = tokens + self.position_embedding + embedding
        else:
            tokens = self.ppg_stem(ppg)
            embedding = self.modality_embedding[1]
            encoded = tokens + self.position_embedding + embedding
        return self.temporal_encoder(self.fusion_norm(encoded))

    def regression_features(self, ecg: torch.Tensor, ppg: torch.Tensor, sqi: torch.Tensor) -> torch.Tensor:
        return self.regressor_hidden(self._active_pool(self._single_tokens(ecg, ppg)))

    def forward(self, ecg: torch.Tensor, ppg: torch.Tensor, sqi: torch.Tensor, *, return_attention: bool = False):
        prediction = self.regressor_out(self.regression_features(ecg, ppg, sqi))
        if return_attention:
            return prediction, {"input_modality": self.modality}
        return prediction


class FusionControlPhysioCAT(PhysioCAT):
    """Matched input encoders with a non-attention fusion operator."""

    def __init__(self, fusion_kind: str, config: PhysioCATConfig | None = None):
        super().__init__(config=config, mask_kind="no_delay")
        if fusion_kind not in {"early_concat", "late_average", "gated", "se"}:
            raise ValueError(fusion_kind)
        self.fusion_kind = fusion_kind
        # These controls compare fusion operators after the same two encoders;
        # unused cross-attention/SQI modules are removed rather than retained
        # as dormant trainable parameters.
        del self.ecg_queries_ppg, self.ppg_queries_ecg
        del self.edge_pair_projection, self.edge_pair_norm, self.edge_dropout
        hidden = self.config.hidden
        if fusion_kind == "early_concat":
            self.early_fusion = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        elif fusion_kind == "gated":
            self.gate = nn.Linear(2 * hidden, hidden)
        elif fusion_kind == "se":
            self.se = nn.Sequential(nn.Linear(2 * hidden, max(16, hidden // 4)), nn.GELU(), nn.Linear(max(16, hidden // 4), 2))

    def fusion_tokens(self, ecg: torch.Tensor, ppg: torch.Tensor, sqi: torch.Tensor, *, return_attention: bool = False):
        e, p = self.encode(ecg, ppg)
        if self.fusion_kind == "early_concat":
            fused = self.early_fusion(torch.cat([e, p], dim=-1))
        elif self.fusion_kind == "late_average":
            fused = 0.5 * (e + p)
        elif self.fusion_kind == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([e, p], dim=-1)))
            fused = gate * e + (1 - gate) * p
        else:
            pooled = torch.cat([e.mean(dim=1), p.mean(dim=1)], dim=-1)
            weights = torch.softmax(self.se(pooled), dim=-1)
            fused = weights[:, 0, None, None] * e + weights[:, 1, None, None] * p
        fused = self.temporal_encoder(self.fusion_norm(fused))
        if return_attention:
            return fused, {"fusion_kind": self.fusion_kind}
        return fused


def build_model(name: str, config: PhysioCATConfig | None = None) -> nn.Module:
    if name == "physiocat":
        return PhysioCAT(config=config, mask_kind="delay_asymmetric")
    if name == "matched_no_delay":
        return MatchedNoDelayPhysioCAT(config=config)
    if name == "physiocat_patch_local":
        return PhysioCAT(config=config, mask_kind="delay_asymmetric")
    if name == "matched_no_delay_patch_local":
        return MatchedNoDelayPhysioCAT(config=config)
    if name == "uniform_delay_band":
        return UniformDelayBandPhysioCAT(config=config)
    if name == "ecg_only":
        return InputAblationPhysioCAT("ecg", config=config)
    if name == "ppg_only":
        return InputAblationPhysioCAT("ppg", config=config)
    if name in {"early_concat", "late_average", "gated_fusion", "se_fusion"}:
        kind = {"gated_fusion": "gated", "se_fusion": "se"}.get(name, name)
        return FusionControlPhysioCAT(kind, config=config)
    delay_variants = {
        "delay_120_300": (120, 300),
        "delay_120_350": (120, 350),
        "delay_80_550": (80, 550),
    }
    if name in delay_variants:
        lower, upper = delay_variants[name]
        return PhysioCAT(config=replace(config or PhysioCATConfig(), lower_ms=lower, upper_ms=upper))
    controls = {
        "ppg_leading_mirror": ("mirror", True),
        "unidirectional_delay": ("unidirectional", True),
        "direction_agnostic_local": ("local", True),
        "shifted_offsets_9_12": ("shifted", True),
        "offset_band_2_5_common": ("offsets_2_5_common", True),
        "offset_band_3_6_common": ("offsets_3_6_common", True),
        "offset_band_4_7_common": ("offsets_4_7_common", True),
        "attention_edge_ablation": ("random", True),
        "without_sqi_fusion": ("delay_asymmetric", False),
        "without_delay_and_sqi": ("no_delay", False),
        "without_pretraining": ("delay_asymmetric", True),
    }
    if name in controls:
        mask_kind, use_sqi = controls[name]
        return PhysioCAT(config=config, mask_kind=mask_kind, use_sqi_fusion=use_sqi)
    raise KeyError(name)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
