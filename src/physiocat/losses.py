from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_bp_loss(prediction: torch.Tensor, target: torch.Tensor, *, order_weight: float = 0.15, huber_delta: float = 5.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    huber = F.huber_loss(prediction, target, delta=huber_delta)
    # Penalize only an actual SBP/DBP inversion.  A positive fixed margin would
    # impose an arbitrary hard lower edge on predicted pulse pressure.
    order = torch.relu(prediction[:, 1] - prediction[:, 0]).mean()
    total = huber + order_weight * order
    return total, {"huber": huber.detach(), "order": order.detach()}


def nt_xent(embeddings_a: torch.Tensor, embeddings_b: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    a = F.normalize(embeddings_a, dim=-1)
    b = F.normalize(embeddings_b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def subject_aware_nt_xent(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    subject_ids: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """InfoNCE with same-window positives and different-subject negatives.

    Off-diagonal windows belonging to the same subject are excluded from the
    denominator instead of being mislabeled as negatives.
    """
    a = F.normalize(embeddings_a, dim=-1)
    b = F.normalize(embeddings_b, dim=-1)
    logits = a @ b.T / temperature
    subject_ids = subject_ids.reshape(-1)
    same_subject = subject_ids[:, None] == subject_ids[None, :]
    diagonal = torch.eye(len(subject_ids), dtype=torch.bool, device=logits.device)
    allowed = ~same_subject | diagonal
    masked = logits.masked_fill(~allowed, -torch.inf)
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(masked, labels) + F.cross_entropy(masked.T, labels))


def subject_aware_multimodal_contrastive_loss(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    subject_ids: torch.Tensor,
    temperature: float = 0.10,
    weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ecg = subject_aware_nt_xent(first["ecg"], second["ecg"], subject_ids, temperature)
    ppg = subject_aware_nt_xent(first["ppg"], second["ppg"], subject_ids, temperature)
    inter_first = subject_aware_nt_xent(first["ecg"], first["ppg"], subject_ids, temperature)
    inter_second = subject_aware_nt_xent(second["ecg"], second["ppg"], subject_ids, temperature)
    inter = 0.5 * (inter_first + inter_second)
    total = weights[0] * ecg + weights[1] * ppg + weights[2] * inter
    return total, {"ecg_intra": ecg.detach(), "ppg_intra": ppg.detach(), "ecg_ppg_inter": inter.detach()}


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
