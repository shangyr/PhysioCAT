from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .losses import gaussian_kl, supervised_bp_loss, subject_aware_multimodal_contrastive_loss


PATCH_LOCAL_INPUT_VIEW = "patch_local"
WINDOW_ROBUST_INPUT_VIEW = "window_robust"
COMMON_WINDOW_ROBUST_MODELS = frozenset(
    {"physiocat", "matched_no_delay", "cnn_bilstm", "bp_net", "te_sagru", "mufubp_net", "random_forest", "pat_ridge"}
)
PATCH_LOCAL_MECHANISM_CONTROLS = frozenset(
    {"physiocat_patch_local", "matched_no_delay_patch_local"}
)


def input_view_for_model(model_name: str) -> str:
    """Return the frozen waveform representation assigned to a model family."""
    if model_name in PATCH_LOCAL_MECHANISM_CONTROLS:
        return PATCH_LOCAL_INPUT_VIEW
    return WINDOW_ROBUST_INPUT_VIEW


def select_waveform_view(archive: dict[str, np.ndarray], input_view: str) -> tuple[np.ndarray, np.ndarray]:
    if input_view == PATCH_LOCAL_INPUT_VIEW:
        ecg_key = "ecg_patch_local" if "ecg_patch_local" in archive else "ecg"
        ppg_key = "ppg_patch_local" if "ppg_patch_local" in archive else "ppg"
    elif input_view == WINDOW_ROBUST_INPUT_VIEW:
        ecg_key, ppg_key = "ecg_window_robust", "ppg_window_robust"
    else:
        raise ValueError(f"Unknown input view: {input_view}")
    missing = [key for key in (ecg_key, ppg_key) if key not in archive]
    if missing:
        raise KeyError(f"Prepared archive lacks the declared {input_view} view: {missing}")
    return archive[ecg_key], archive[ppg_key]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def initialize_model(model_factory: Callable[[], nn.Module], seed: int) -> nn.Module:
    """Construct a model only after applying its registered initialization seed."""
    if not callable(model_factory):
        raise TypeError("model_factory must be a zero-argument callable so initialization can be seeded")
    seed_everything(seed)
    model = model_factory()
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return torch.nn.Module")
    return model


def subject_group_code(value: str) -> int:
    """Stable integer group label; no target or fold information is encoded."""
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**63 - 1)


class WaveformDataset(Dataset):
    def __init__(
        self,
        archive: dict[str, np.ndarray],
        indices: np.ndarray | None = None,
        *,
        input_view: str = PATCH_LOCAL_INPUT_VIEW,
    ):
        self.archive = archive
        self.ecg, self.ppg = select_waveform_view(archive, input_view)
        self.input_view = input_view
        self.indices = np.arange(len(self.ecg)) if indices is None else np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        subject = str(self.archive["subject_id"][index])
        item_data = {
            "ecg": torch.from_numpy(self.ecg[index]).float().unsqueeze(0),
            "ppg": torch.from_numpy(self.ppg[index]).float().unsqueeze(0),
            "sqi": torch.from_numpy(self.archive["sqi_tokens"][index]).float(),
            "target": torch.from_numpy(self.archive["targets"][index]).float(),
            "subject_id": subject,
            "subject_code": torch.tensor(subject_group_code(subject), dtype=torch.long),
            "window_id": str(self.archive["window_id"][index]),
        }
        return item_data


def _paired_roll(signal: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Apply the same temporal shift to ECG and PPG for each paired view."""
    rows = []
    for row, shift in zip(signal, shifts.tolist()):
        rows.append(torch.roll(row, int(shift), dims=-1))
    return torch.stack(rows, dim=0)


def paired_augment(ecg: torch.Tensor, ppg: torch.Tensor, *, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    batch = ecg.shape[0]
    shifts = torch.randint(-8, 9, (batch,), generator=generator, device=ecg.device)
    scale_e = 1.0 + 0.04 * torch.randn((batch, 1, 1), generator=generator, device=ecg.device)
    scale_p = 1.0 + 0.04 * torch.randn((batch, 1, 1), generator=generator, device=ecg.device)
    e = _paired_roll(scale_e * ecg + 0.025 * torch.randn_like(ecg), shifts)
    p = _paired_roll(scale_p * ppg + 0.025 * torch.randn_like(ppg), shifts)
    return e, p


def prediction_features(model: nn.Module, ecg, ppg, sqi):
    if hasattr(model, "regression_features"):
        return model.regression_features(ecg, ppg, sqi)
    raise TypeError("Model does not expose regression_features")


class ProjectionHeads(nn.Module):
    """Training-only modality heads; saved separately and discarded for inference."""
    def __init__(self, dimensions: dict[str, int], projection_dim: int = 128):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Linear(dim, projection_dim),
                )
                for name, dim in dimensions.items()
            }
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.heads[name](value) for name, value in features.items()}


def pretrain_epoch(
    model,
    projectors,
    loader,
    optimizer,
    device,
    temperature: float = 0.10,
    component_weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
):
    model.train(); projectors.train()
    total = 0.0
    for batch in loader:
        ecg, ppg, sqi = batch["ecg"].to(device), batch["ppg"].to(device), batch["sqi"].to(device)
        ecg1, ppg1 = paired_augment(ecg, ppg)
        ecg2, ppg2 = paired_augment(ecg, ppg)
        first = projectors(model.contrastive_features(ecg1, ppg1, sqi))
        second = projectors(model.contrastive_features(ecg2, ppg2, sqi))
        subject_ids = batch["subject_code"].to(device)
        loss, _ = subject_aware_multimodal_contrastive_loss(
            first, second, subject_ids, temperature, component_weights
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projectors.parameters()), 5.0)
        optimizer.step()
        total += float(loss.detach()) * len(ecg)
    return total / max(1, len(loader.dataset))


def supervised_epoch(model, loader, optimizer, device):
    model.train(); total = 0.0; components = {"bp": 0.0, "kl": 0.0}
    for batch in loader:
        ecg, ppg, sqi, target = (batch[k].to(device) for k in ("ecg", "ppg", "sqi", "target"))
        if hasattr(model, "forward_with_aux"):
            prediction, aux = model.forward_with_aux(ecg, ppg)
        else:
            prediction, aux = model(ecg, ppg, sqi), {}
        loss, detail = supervised_bp_loss(prediction, target)
        if "mu" in aux and "logvar" in aux:
            kl = gaussian_kl(aux["mu"], aux["logvar"])
            loss = loss + 1e-3 * kl
            components["kl"] += float(kl.detach()) * len(ecg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach()) * len(ecg)
        components["bp"] += float(detail["huber"]) * len(ecg)
    n = max(1, len(loader.dataset))
    return total / n, {key: value / n for key, value in components.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); predictions, targets, windows, subjects = [], [], [], []
    for batch in loader:
        pred = model(batch["ecg"].to(device), batch["ppg"].to(device), batch["sqi"].to(device)).cpu()
        predictions.append(pred); targets.append(batch["target"])
        windows.extend(batch["window_id"]); subjects.extend(batch["subject_id"])
    prediction = torch.cat(predictions).numpy(); target = torch.cat(targets).numpy()
    mae = np.mean(np.abs(prediction - target), axis=0)
    return {"prediction": prediction, "target": target, "window_id": windows, "subject_id": subjects,
            "sbp_mae": float(mae[0]), "dbp_mae": float(mae[1]), "objective": float(mae.mean())}


@dataclass
class FitConfig:
    seed: int = 42
    data_order_seed: int | None = None
    batch_size: int = 32
    pretrain_epochs: int = 5
    maximum_epochs: int = 80
    patience: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    pretrain_enabled: bool | None = None
    contrastive_temperature: float = 0.10
    contrastive_projection_dim: int = 128
    contrastive_component_weights: tuple[float, float, float] = (1.0, 1.0, 0.5)
    num_workers: int = 0


def _cosine_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + np.cos(np.pi * np.clip(progress, 0.0, 1.0)))


def fit_model(model_factory: Callable[[], nn.Module], train_dataset, validation_dataset, output_dir: Path, config: FitConfig | None = None, device: str | None = None):
    c = config or FitConfig()
    model = initialize_model(model_factory, c.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); model = model.to(device)
    loader_generator = torch.Generator().manual_seed(int(c.seed if c.data_order_seed is None else c.data_order_seed))
    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers, generator=loader_generator)
    validation_loader = DataLoader(validation_dataset, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers)
    do_pretrain = bool(c.pretrain_epochs > 0 and (c.pretrain_enabled is not False) and hasattr(model, "contrastive_features"))
    projectors = None
    if do_pretrain:
        hidden = int(getattr(getattr(model, "config", None), "hidden", 128))
        projectors = ProjectionHeads(
            {"ecg": hidden, "ppg": hidden}, projection_dim=c.contrastive_projection_dim
        ).to(device)
    history = []; global_epoch = 0
    if do_pretrain:
        # Contrastive pre-training and supervised regression are separate
        # optimization phases.  Each phase starts from a fresh AdamW state and
        # its own warm-up/cosine schedule; only the encoder weights cross the
        # phase boundary.  This is also the schedule recorded in the released
        # selected-to-stop training histories.
        pretrain_parameters = list(model.parameters()) + list(projectors.parameters())
        pretrain_optimizer = torch.optim.AdamW(
            pretrain_parameters, lr=c.learning_rate, weight_decay=c.weight_decay
        )
        pretrain_scheduler = torch.optim.lr_scheduler.LambdaLR(
            pretrain_optimizer,
            lambda step: _cosine_warmup(step, max(1, c.pretrain_epochs), c.warmup_epochs),
        )
        for epoch in range(c.pretrain_epochs):
            epoch_learning_rate = float(pretrain_optimizer.param_groups[0]["lr"])
            loss = pretrain_epoch(
                model,
                projectors,
                train_loader,
                pretrain_optimizer,
                device,
                c.contrastive_temperature,
                c.contrastive_component_weights,
            )
            pretrain_scheduler.step(); global_epoch += 1
            history.append({
                "stage": "contrastive",
                "epoch": epoch + 1,
                "loss": loss,
                "learning_rate": epoch_learning_rate,
                "optimizer_phase": "contrastive",
                "scheduler_phase_epoch": epoch + 1,
            })
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "projection_state_dict": projectors.state_dict(),
            "optimizer_state_dict": pretrain_optimizer.state_dict(),
            "scheduler_state_dict": pretrain_scheduler.state_dict(),
            "optimizer_phase": "contrastive",
            "fit_config": asdict(c),
        }, output_dir / "contrastive_checkpoint.pt")
    supervised_optimizer = torch.optim.AdamW(
        model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay
    )
    supervised_scheduler = torch.optim.lr_scheduler.LambdaLR(
        supervised_optimizer,
        lambda step: _cosine_warmup(step, max(1, c.maximum_epochs), c.warmup_epochs),
    )
    best_state, best_objective, stale = None, float("inf"), 0
    for epoch in range(c.maximum_epochs):
        epoch_learning_rate = float(supervised_optimizer.param_groups[0]["lr"])
        loss, details = supervised_epoch(model, train_loader, supervised_optimizer, device)
        validation = evaluate(model, validation_loader, device)
        supervised_scheduler.step(); global_epoch += 1
        history.append({
            "stage": "supervised",
            "epoch": epoch + 1,
            "loss": loss,
            **details,
            "validation_sbp_mae": validation["sbp_mae"],
            "validation_dbp_mae": validation["dbp_mae"],
            "validation_objective": validation["objective"],
            "learning_rate": epoch_learning_rate,
            "optimizer_phase": "supervised",
            "scheduler_phase_epoch": epoch + 1,
        })
        if validation["objective"] < best_objective - 1e-5:
            best_objective = validation["objective"]; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= c.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state); output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "fit_config": asdict(c),
        "best_validation_objective": best_objective,
        "architecture": type(model).__name__,
        "optimizer_state_dict": supervised_optimizer.state_dict(),
        "scheduler_state_dict": supervised_scheduler.state_dict(),
        "optimizer_phase": "supervised",
        "contrastive_enabled": do_pretrain,
        "global_epochs": global_epoch,
    }
    torch.save(checkpoint, output_dir / "best_checkpoint.pt")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history


def subject_partition(
    subject_ids: np.ndarray,
    test_subjects: str | set[str] | list[str] | np.ndarray,
    validation_subjects: set[str],
):
    subject_ids = np.asarray(subject_ids).astype(str)
    if isinstance(test_subjects, str):
        test_subjects = {test_subjects}
    else:
        test_subjects = set(np.asarray(list(test_subjects)).astype(str))
    test = np.isin(subject_ids, list(test_subjects))
    validation = np.isin(subject_ids, list(validation_subjects))
    train = ~(test | validation)
    if np.any(test & validation) or np.any(test & train) or np.any(validation & train):
        raise AssertionError("Subject partition overlap")
    if not test.any() or not validation.any() or not train.any():
        raise AssertionError("Every subject partition must contain train, validation, and test rows")
    return np.flatnonzero(train), np.flatnonzero(validation), np.flatnonzero(test)
