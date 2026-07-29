from pathlib import Path
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from physiocat.baselines import EngineeredRandomForest, PATRidge, build_neural_baseline, pat_ridge_features
from physiocat.checkpointing import load_inference_checkpoint, save_inference_checkpoint
from physiocat.adapters import _abp_targets, _candidate_failure_code, export_dataset
from physiocat.dataio import iter_hdf5_records
from physiocat.models import MatchedNoDelayPhysioCAT, PhysioCAT, PhysioCATConfig, SafeCrossAttentionBranch, active_query_rows, build_model, delay_mask, mask_row_audit, mechanism_mask, parameter_count
from physiocat.preprocessing import PreprocessingConfig, abp_beat_labels, aligned_abp_scalar_targets, ecg_sqi, patch_center_positions, patch_local_normalize, ppg_sqi, preprocess_window, retention_decision, robust_normalize, synchronization_stability, token_sqi
from physiocat.perturbations import cross_subject_ppg_pairing, token_aligned_circular_shift_ppg, reverse_ppg
from physiocat.losses import supervised_bp_loss, subject_aware_multimodal_contrastive_loss, subject_aware_nt_xent, gaussian_kl
from physiocat.statistics import holm_adjust
from physiocat.training import FitConfig, ProjectionHeads, WaveformDataset, initialize_model, input_view_for_model, paired_augment, _cosine_warmup
from canonicalize_candidate_lineage import OUTPUT_COLUMNS, canonicalize_candidate_audit


def tiny_config():
    return PhysioCATConfig(hidden=32, heads=4, ffn_multiplier=2, patch_mlp_multiplier=2, temporal_layers=1, dropout=0.0)


def qrs_test_fixture(t: np.ndarray, rate_hz: float) -> np.ndarray:
    """Return a deterministic QRS-like fixture suitable for R-peak tests."""
    phase = np.mod(np.asarray(t, dtype=np.float64) * rate_hz, 1.0)
    return (
        1.20 * np.exp(-0.5 * ((phase - 0.10) / 0.015) ** 2)
        - 0.25 * np.exp(-0.5 * ((phase - 0.07) / 0.012) ** 2)
        - 0.20 * np.exp(-0.5 * ((phase - 0.14) / 0.018) ** 2)
        + 0.03 * np.sin(2 * np.pi * rate_hz * np.asarray(t, dtype=np.float64))
    )


def test_initialization_seed_is_applied_before_model_construction():
    torch.manual_seed(9001)
    first = initialize_model(lambda: PhysioCAT(tiny_config()), 42)
    torch.manual_seed(17)
    second = initialize_model(lambda: PhysioCAT(tiny_config()), 42)
    third = initialize_model(lambda: PhysioCAT(tiny_config()), 43)
    first_state = first.state_dict()
    second_state = second.state_dict()
    third_state = third.state_dict()
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)
    assert any(not torch.equal(first_state[name], third_state[name]) for name in first_state)


def test_contrastive_projection_and_component_weight_contract():
    model = PhysioCAT(tiny_config()).eval()
    features = model.contrastive_features(
        torch.randn(3, 1, 2000), torch.randn(3, 1, 2000), torch.rand(3, 2, 125)
    )
    assert set(features) == {"ecg", "ppg"}
    projectors = ProjectionHeads({"ecg": 32, "ppg": 32})
    projected = projectors(features)
    assert set(projected) == {"ecg", "ppg"}
    assert projected["ecg"].shape == projected["ppg"].shape == (3, 128)
    config = FitConfig()
    assert config.contrastive_projection_dim == 128
    assert config.contrastive_component_weights == (1.0, 1.0, 0.5)
    second = {name: value.roll(1, dims=0) for name, value in projected.items()}
    subjects = torch.arange(3)
    total, parts = subject_aware_multimodal_contrastive_loss(projected, second, subjects)
    expected = parts["ecg_intra"] + parts["ppg_intra"] + 0.5 * parts["ecg_ppg_inter"]
    assert torch.allclose(total.detach(), expected)


def test_waveform_perturbations_are_deterministic_and_respect_their_contracts():
    ppg = np.arange(6 * 32, dtype=float).reshape(6, 1, 32)
    subjects = np.asarray(["s1", "s1", "s2", "s2", "s3", "s3"])
    shifted_a, shifts_a = token_aligned_circular_shift_ppg(
        ppg, minimum_shift_tokens=1, maximum_shift_tokens=3, patch_samples=4, seed=120450
    )
    shifted_b, shifts_b = token_aligned_circular_shift_ppg(
        ppg, minimum_shift_tokens=1, maximum_shift_tokens=3, patch_samples=4, seed=120450
    )
    assert np.array_equal(shifted_a, shifted_b) and np.array_equal(shifts_a, shifts_b)
    assert np.all(np.abs(shifts_a) >= 4) and np.all(np.abs(shifts_a) <= 12)
    assert np.all(shifts_a % 4 == 0)
    assert np.array_equal(np.sort(shifted_a, axis=-1), np.sort(ppg, axis=-1))
    assert np.allclose(np.abs(np.fft.rfft(shifted_a, axis=-1)), np.abs(np.fft.rfft(ppg, axis=-1)))
    original_tokens = ppg.reshape(6, 1, 8, 4)
    shifted_tokens = shifted_a.reshape(6, 1, 8, 4)
    for row in range(len(ppg)):
        assert {tuple(token) for token in original_tokens[row, 0]} == {tuple(token) for token in shifted_tokens[row, 0]}
    paired, donor_index = cross_subject_ppg_pairing(ppg, subjects, seed=2714)
    assert np.array_equal(paired, ppg[donor_index])
    assert np.all(subjects != subjects[donor_index])
    assert np.array_equal(reverse_ppg(ppg), ppg[..., ::-1])
    assert np.array_equal(ppg, np.arange(6 * 32, dtype=float).reshape(6, 1, 32))
    import yaml
    protocol = pd.read_csv(ROOT / "artifacts/protocol/timing_perturbation_protocol.csv")
    phase = protocol[protocol.control == "token_aligned_ppg_shift"].iloc[0]
    config = yaml.safe_load((ROOT / "configs/experiments/mechanism_controls.yaml").read_text(encoding="utf-8"))
    frozen = config["controls"]["timing_perturbations"]["token_aligned_ppg_shift"]
    assert int(phase.seed) == int(frozen["seed"]) == 160000
    assert int(phase.minimum_absolute_shift_tokens) == int(frozen["minimum_absolute_shift_tokens"]) == 3
    assert int(phase.maximum_absolute_shift_tokens) == int(frozen["maximum_absolute_shift_tokens"]) == 10
    assert int(phase.patch_samples) == int(frozen["patch_samples"]) == 16
    assert int(phase.minimum_absolute_shift_ms) == 192 and int(phase.maximum_absolute_shift_ms) == 640
    long_ppg = np.arange(4 * 2000, dtype=float).reshape(4, 1, 2000)
    shifted, realized = token_aligned_circular_shift_ppg(
        long_ppg,
        minimum_shift_tokens=int(phase.minimum_absolute_shift_tokens),
        maximum_shift_tokens=int(phase.maximum_absolute_shift_tokens),
        patch_samples=int(phase.patch_samples),
        seed=int(phase.seed),
    )
    assert np.all(np.abs(realized) >= 48) and np.all(np.abs(realized) <= 160)
    assert np.all(realized % 16 == 0)
    assert np.allclose(np.abs(np.fft.rfft(shifted, axis=-1)), np.abs(np.fft.rfft(long_ppg, axis=-1)))


def test_delay_mask_has_only_ecg_leading_pairs():
    mask = delay_mask()
    row, col = torch.where(mask)
    offsets = col - row
    token_center_delay = offsets * 64
    raw_support_low = token_center_delay - 60
    raw_support_high = token_center_delay + 60
    assert int(raw_support_low.min()) >= 120
    assert int(raw_support_high.max()) <= 450


def test_both_delay_branches_encode_ecg_leading_ppg_relation():
    forward_row, forward_col = torch.where(delay_mask())
    reverse_row, reverse_col = torch.where(delay_mask(reverse=True))
    assert torch.equal(torch.unique(forward_col - forward_row), torch.arange(3, 7))
    assert torch.equal(torch.unique(reverse_row - reverse_col), torch.arange(3, 7))
    assert int(delay_mask().sum()) == int(delay_mask(reverse=True).sum()) == 482


def test_ppg_leading_mirror_reverses_only_the_delay_direction():
    forward = mechanism_mask("mirror")
    reverse = mechanism_mask("mirror", reverse=True)
    forward_row, forward_col = torch.where(forward)
    reverse_row, reverse_col = torch.where(reverse)
    assert torch.equal(torch.unique(forward_col - forward_row), torch.arange(-6, -2))
    assert torch.equal(torch.unique(reverse_col - reverse_row), torch.arange(3, 7))
    assert torch.equal(forward, reverse.transpose(0, 1))
    assert int(forward.sum()) == int(reverse.sum()) == 482


def test_equal_width_location_sweep_has_identical_edge_capacity_and_support():
    for kind in (
        "offsets_2_5_common",
        "offsets_3_6_common",
        "offsets_4_7_common",
        "offsets_9_12_common",
    ):
        forward = mechanism_mask(kind)
        reverse = mechanism_mask(kind, reverse=True).transpose(0, 1)
        assert torch.equal(forward, reverse)
        assert int(forward.sum()) == 452
        row, _ = torch.where(forward)
        assert int(row.min()) == 0 and int(row.max()) == 112


def test_empty_attention_rows_are_exactly_zero():
    branch = SafeCrossAttentionBranch(hidden=32, heads=4, dropout=0.0).eval()
    query = torch.randn(2, 5, 32)
    key = torch.randn(2, 5, 32)
    allowed = torch.zeros(5, 5, dtype=torch.bool)
    allowed[:3, :2] = True
    output, weights = branch(query, key, allowed)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert torch.equal(output[:, 3:], torch.zeros_like(output[:, 3:]))
    assert torch.equal(weights[:, :, 3:], torch.zeros_like(weights[:, :, 3:]))


def test_reciprocal_attention_scores_are_fused_on_the_same_ecg_ppg_edges():
    model = PhysioCAT(tiny_config()).eval()
    ecg = torch.randn(1, 1, 2000)
    ppg = torch.randn(1, 1, 2000)
    sqi = torch.full((1, 2, 125), 0.5)
    with torch.no_grad():
        pair_message, attention = model.edge_aligned_pair_message(ecg, ppg, sqi)
        _, base_attention = model.pre_temporal_fusion(ecg, ppg, sqi)
        changed_sqi = sqi.clone(); changed_sqi[:, 0, 0] = 1.0
        _, changed_attention = model.pre_temporal_fusion(ecg, ppg, changed_sqi)
    assert pair_message.shape == (1, 125, 32)
    assert torch.equal(pair_message[:, -3:], torch.zeros_like(pair_message[:, -3:]))
    allowed = delay_mask()
    forward_support = attention["ecg_query_ppg_key_value"].sum(dim=1)[0] > 0
    reverse_support = attention["ppg_query_ecg_key_value"].sum(dim=1)[0].transpose(0, 1) > 0
    pair_support = attention["edge_aligned_pair"].sum(dim=1)[0] > 0
    assert torch.equal(forward_support, allowed)
    assert torch.equal(reverse_support, allowed)
    assert torch.equal(pair_support, allowed)
    assert not torch.equal(base_attention["edge_anchor_reliability"][:, 0], changed_attention["edge_anchor_reliability"][:, 0])
    assert torch.equal(base_attention["edge_anchor_reliability"][:, 1:], changed_attention["edge_anchor_reliability"][:, 1:])


def test_boundary_rows_are_masked_from_temporal_context_and_pooling():
    model = PhysioCAT(tiny_config()).eval()
    with torch.no_grad():
        fused, attention = model.fusion_tokens(
            torch.randn(2, 1, 2000), torch.randn(2, 1, 2000), torch.rand(2, 2, 125), return_attention=True
        )
    assert attention["active_rows"].sum() == 122
    assert torch.equal(fused[:, -3:], torch.zeros_like(fused[:, -3:]))


def test_matched_no_delay_has_identical_parameter_count():
    config = tiny_config()
    assert parameter_count(PhysioCAT(config)) == parameter_count(MatchedNoDelayPhysioCAT(config))


def test_full_architecture_forward_and_attention_shapes():
    model = PhysioCAT(tiny_config()).eval()
    ecg = torch.randn(2, 1, 2000)
    ppg = torch.randn(2, 1, 2000)
    sqi = torch.rand(2, 2, 125)
    prediction, attention = model(ecg, ppg, sqi, return_attention=True)
    assert prediction.shape == (2, 2)
    assert attention["ecg_query_ppg_key_value"].shape == (2, 4, 125, 125)
    assert attention["ppg_query_ecg_key_value"].shape == (2, 4, 125, 125)
    assert attention["edge_aligned_pair"].shape == (2, 4, 125, 125)
    assert attention["edge_pair_reliability"].shape == (2, 125, 125)
    assert attention["edge_anchor_reliability"].shape == (2, 125)
    assert "ecg_query_incoming_mass_on_ppg_timeline" not in attention
    assert "aligned_ecg_query_sqi_on_ppg_timeline" not in attention


def test_uniform_delay_band_control_uses_fixed_equal_affinity_on_the_same_edges():
    model = build_model("uniform_delay_band", tiny_config()).eval()
    with torch.no_grad():
        _, attention = model(
            torch.randn(2, 1, 2000), torch.randn(2, 1, 2000), torch.ones(2, 2, 125),
            return_attention=True,
        )
    weights = attention["ecg_query_ppg_key_value"]
    allowed = delay_mask(n_tokens=125)
    for row in torch.where(allowed.any(dim=1))[0]:
        values = weights[:, :, row, allowed[row]]
        assert torch.allclose(values, torch.full_like(values, 1.0 / int(allowed[row].sum())))
    assert not hasattr(model.ecg_queries_ppg, "q") and not hasattr(model.ecg_queries_ppg, "k")


def test_joint_fusion_has_no_unmasked_raw_modality_residual_bypass():
    model = PhysioCAT(tiny_config(), use_sqi_fusion=False).eval()
    first_ecg, first_ppg = torch.randn(2, 1, 2000), torch.randn(2, 1, 2000)
    second_ecg, second_ppg = torch.randn(2, 1, 2000), torch.randn(2, 1, 2000)
    sqi = torch.rand(2, 2, 125)
    with torch.no_grad():
        model.edge_pair_projection.weight.zero_()
        first, _ = model.pre_temporal_fusion(first_ecg, first_ppg, sqi)
        second, _ = model.pre_temporal_fusion(second_ecg, second_ppg, sqi)
    assert torch.equal(first, second)


def test_input_and_fusion_controls_have_no_dormant_cross_modal_access():
    ecg_only = build_model("ecg_only", tiny_config())
    ppg_only = build_model("ppg_only", tiny_config())
    assert not hasattr(ecg_only, "ppg_stem") and not hasattr(ecg_only, "ecg_queries_ppg")
    assert not hasattr(ppg_only, "ecg_stem") and not hasattr(ppg_only, "ppg_queries_ecg")
    for name in ("early_concat", "late_average", "gated_fusion", "se_fusion"):
        model = build_model(name, tiny_config())
        assert not hasattr(model, "ecg_queries_ppg") and not hasattr(model, "edge_pair_projection")
        with torch.no_grad():
            output = model(torch.randn(1, 1, 2000), torch.randn(1, 1, 2000), torch.rand(1, 2, 125))
        assert output.shape == (1, 2) and torch.isfinite(output).all()


def test_fusion_controls_retain_the_matched_post_fusion_temporal_encoder():
    for name in ("early_concat", "late_average", "gated_fusion", "se_fusion"):
        model = build_model(name, tiny_config()).eval()
        calls = []
        hook = model.temporal_encoder.register_forward_hook(lambda *_: calls.append(True))
        try:
            with torch.no_grad():
                model(torch.randn(1, 1, 2000), torch.randn(1, 1, 2000), torch.rand(1, 2, 125))
        finally:
            hook.remove()
        assert calls == [True]


def test_patch_stem_and_encoder_have_strict_bounded_history():
    config = tiny_config()
    model = PhysioCAT(config).eval()
    assert model.ecg_stem.receptive_field_samples == 16
    ecg = torch.randn(1, 1, 2000)
    ppg = torch.randn(1, 1, 2000)
    target_token = 50
    changed = ecg.clone()
    changed[:, :, 47 * 16:48 * 16] += 100.0
    with torch.no_grad():
        original, _ = model.encode(ecg, ppg)
        perturbed, _ = model.encode(changed, ppg)
    assert torch.equal(original[:, target_token], perturbed[:, target_token])
    assert model.ecg_stem.patch_projection.kernel_size == (16,)
    assert model.ecg_stem.patch_projection.stride == (16,)
    assert len(model.temporal_encoder.layers) == config.temporal_layers


def test_patch_local_sensitivity_view_is_strictly_local_on_the_analysis_grid():
    rng = np.random.default_rng(120450)
    ecg = rng.normal(size=2000)
    ppg = rng.normal(size=2000)
    target_patch = 50
    changed_ecg = ecg.copy()
    changed_ppg = ppg.copy()
    changed_ecg[47 * 16:48 * 16] += 100.0
    changed_ppg[82 * 16:83 * 16] -= 100.0
    original_ecg = patch_local_normalize(ecg)
    original_ppg = patch_local_normalize(ppg)
    perturbed_ecg = patch_local_normalize(changed_ecg)
    perturbed_ppg = patch_local_normalize(changed_ppg)
    target = slice(target_patch * 16, (target_patch + 1) * 16)
    assert np.array_equal(original_ecg[target], perturbed_ecg[target])
    assert np.array_equal(original_ppg[target], perturbed_ppg[target])


def test_token_sqi_is_interpolated_to_exact_patch_centers():
    centers = patch_center_positions(2000, 125)
    assert np.array_equal(centers[:3], np.array([7.5, 23.5, 39.5]))
    assert centers[-1] == 1991.5
    assert np.all(np.diff(centers) == 16.0)


def test_preprocess_window_keeps_analysis_filtering_out_of_model_waveforms():
    t = np.arange(2000) / 250.0
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 18 * t)
    ppg = 2.0 + np.sin(2 * np.pi * 1.2 * (t - 0.24))
    result = preprocess_window(ecg, ppg, 250, age_years=60, released_sbp=122, released_dbp=72)
    assert np.array_equal(result["ecg"], robust_normalize(ecg))
    assert np.array_equal(result["ppg"], robust_normalize(ppg))
    assert np.array_equal(result["ecg_patch_local"], patch_local_normalize(ecg))
    assert np.array_equal(result["ppg_patch_local"], patch_local_normalize(ppg))
    assert np.array_equal(result["ecg_window_robust"], robust_normalize(ecg))
    assert np.array_equal(result["ppg_window_robust"], robust_normalize(ppg))


def test_model_families_receive_their_prespecified_input_views():
    archive = {
        "ecg": np.zeros((2, 2000), dtype=np.float32),
        "ppg": np.zeros((2, 2000), dtype=np.float32),
        "ecg_patch_local": np.full((2, 2000), 1.0, dtype=np.float32),
        "ppg_patch_local": np.full((2, 2000), 2.0, dtype=np.float32),
        "ecg_window_robust": np.full((2, 2000), 3.0, dtype=np.float32),
        "ppg_window_robust": np.full((2, 2000), 4.0, dtype=np.float32),
        "sqi_tokens": np.ones((2, 2, 125), dtype=np.float32),
        "targets": np.zeros((2, 2), dtype=np.float32),
        "subject_id": np.asarray(["S1", "S2"]),
        "window_id": np.asarray(["W1", "W2"]),
    }
    proposed = WaveformDataset(archive, input_view=input_view_for_model("physiocat"))[0]
    baseline = WaveformDataset(archive, input_view=input_view_for_model("mufubp_net"))[0]
    assert torch.all(proposed["ecg"] == 3.0) and torch.all(proposed["ppg"] == 4.0)
    assert torch.all(baseline["ecg"] == 3.0) and torch.all(baseline["ppg"] == 4.0)
    assert input_view_for_model("matched_no_delay") == "window_robust"
    assert input_view_for_model("physiocat_patch_local") == "patch_local"
    assert input_view_for_model("matched_no_delay_patch_local") == "patch_local"
    assert all(input_view_for_model(name) == "window_robust" for name in ("cnn_bilstm", "bp_net", "te_sagru", "mufubp_net", "random_forest", "pat_ridge"))


def test_retention_does_not_read_pat():
    record = {name: True for name in ("adult_metadata", "scalar_target_valid", "beat_count_pass", "ecg_quality_pass", "ppg_quality_pass", "paired_sample_continuity", "sqi_rule_pass")}
    record.update({"pat_detected": False, "pat_ms": 649, "pat_in_model_band": False, "abp_quality_pass": False, "reference_valid": False})
    assert retention_decision(record)


def test_external_cohort_registry_uses_one_fixed_mimic_bp_view():
    registry = pd.read_csv(ROOT / "artifacts/cohorts/cohort_registry.csv").set_index("cohort")
    predictions = pd.read_csv(ROOT / "artifacts/predictions/mimic_bp_zero_shot_predictions.csv.gz")
    assert int(registry.loc["mimic_bp", "subjects"]) == 1524
    assert int(registry.loc["mimic_bp", "windows"]) == len(predictions)
    assert predictions.subject_id.nunique() == 1524
    manuscript = (ROOT / "paper/PhysioCAT_Manuscript.tex").read_text(encoding="utf-8")
    assert "independent external target cohort" not in manuscript
    assert "common identity audit" not in manuscript


def test_mimic_target_protocol_identity_audit_is_explicit():
    audit = pd.read_csv(ROOT / "artifacts/cohorts/mimic_ecosystem_identity_audit.csv").iloc[0]
    assert audit.cohort_a == "pulsedb_mimic" and audit.cohort_b == "mimic_bp"
    assert int(audit.shared_source_patient_hashes) == 0
    assert int(audit.shared_source_record_hashes) == 0
    assert "patient-disjoint retained target protocols" in audit.interpretation
    assert "not independent clinical sites" in audit.interpretation


def test_raw_candidate_lineage_is_complete_and_has_no_hidden_record_cap():
    expected = {
        "pulsedb_vital": (240_428, 225_536),
        "pulsedb_mimic": (211_414, 198_800),
        "mimic_bp": (47_059, 42_822),
    }
    prediction_names = {
        "pulsedb_vital": "pulsedb_vital_subject_grouped_oof_predictions.csv.gz",
        "pulsedb_mimic": "pulsedb_mimic_zero_shot_predictions.csv.gz",
        "mimic_bp": "mimic_bp_zero_shot_predictions.csv.gz",
    }
    public_columns = [
        "cohort", "raw_candidate_id", "subject_id", "source_record_hash",
        "age_years", "sex", "repository_sbp", "repository_dbp",
        "repository_scalar_available", "input_min_sqi", "rr_irregularity",
        "paired_waveform_crop_pass", "target_derivation_pass",
        "target_failure_reason", "status", "window_id",
    ]
    for slug, (raw_rows, target_rows) in expected.items():
        raw = pd.read_csv(ROOT / f"data/retention/{slug}_raw_candidate_manifest.csv.gz", keep_default_na=False)
        target = pd.read_csv(ROOT / f"data/retention/{slug}_window_audit.csv.gz")
        prediction = pd.read_csv(ROOT / "artifacts/predictions" / prediction_names[slug])
        assert len(raw) == raw_rows and len(target) == target_rows
        assert raw.source_record_hash.is_unique
        assert list(raw.columns) == public_columns
        accepted = raw.loc[raw.target_derivation_pass.astype(bool), "window_id"]
        assert accepted.is_unique and set(accepted) == set(target.window_id)
        assert prediction.source_record_hash.is_unique


def test_post_target_retention_has_no_narrow_bp_outcome_range_screen():
    for slug in ("pulsedb_vital", "pulsedb_mimic", "mimic_bp"):
        config = yaml.safe_load((ROOT / f"configs/data/{slug}.yaml").read_text(encoding="utf-8"))
        assert config["labels"]["outcome_range_used_for_post_target_retention"] is False
        assert "scalar_target_valid" in config["retention"]["required_checks"]
    manuscript = (ROOT / "paper/PhysioCAT_Manuscript.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "paper/PhysioCAT_Supplementary_Material.tex").read_text(encoding="utf-8")
    assert "at most one deterministic centered crop per source-listed segment" in manuscript
    assert "crop and target-formation outcomes are retained in the public reproducibility repository" in supplement.lower()
    assert "not an outcome-range screen" not in manuscript


def test_scalar_target_and_loss_contracts_do_not_use_narrow_bp_ranges():
    t = np.linspace(0, 8, 2000, endpoint=False)
    r_peaks = np.arange(100, 1900, 200)
    ecg = qrs_test_fixture(t, 1.25)
    high_abp = 190.0 + 55.0 * np.sin(2 * np.pi * 1.25 * (t - 0.1))
    sbp, dbp, _, _ = abp_beat_labels(high_abp, r_peaks, fs=250)
    assert sbp > 240 and dbp > 130
    canonical_sbp, canonical_dbp, _, _ = aligned_abp_scalar_targets(ecg, high_abp, fs=250)
    adapter_sbp, adapter_dbp = _abp_targets(ecg, high_abp, 250)
    assert canonical_sbp > 240 and canonical_dbp > 130
    assert adapter_sbp == canonical_sbp and adapter_dbp == canonical_dbp
    adapter_source = (ROOT / "src/physiocat/adapters.py").read_text(encoding="utf-8")
    training_source = (ROOT / "src/physiocat/training.py").read_text(encoding="utf-8")
    assert "aligned_abp_scalar_targets(ecg, abp" in adapter_source and "find_peaks(abp" not in adapter_source
    assert "warn_only=True" not in training_source
    loss, parts = supervised_bp_loss(torch.tensor([[245.0, 135.0]]), torch.tensor([[245.0, 135.0]]))
    assert set(parts) == {"huber", "order"} and torch.isfinite(loss)


def test_order_penalty_prevents_inversion_without_a_fixed_pulse_pressure_floor():
    valid = torch.tensor([[100.0, 99.0]])
    inverted = torch.tensor([[99.0, 100.0]])
    _, valid_parts = supervised_bp_loss(valid, valid)
    _, inverted_parts = supervised_bp_loss(inverted, inverted)
    assert float(valid_parts["order"]) == 0.0
    assert float(inverted_parts["order"]) == 1.0


def test_supplement_has_no_row_break_before_literal_percent_sign():
    supplement = (ROOT / "paper/PhysioCAT_Supplementary_Material.tex").read_text(encoding="utf-8")
    assert r"\\%" not in supplement
    assert r"\label{tab:training_hyperparameters}" in supplement
    assert "reciprocal affinities and pairwise ECG/PPG SQI are fused once on each admissible edge" in supplement
    assert "Reproducibility summary for the subject-disjoint benchmark" in supplement
    assert "Supporting repository record" in supplement
    assert "Prediction-authority and released-checkpoint integrity manifests are provided in the public reproducibility repository" not in supplement
    assert "Formal training-run ledger" not in supplement
    assert "GPU-hours" not in supplement
    assert "271 fixed-seed subjects from the four non-test groups form the validation set" in supplement
    assert "Zero-phase fourth-order Butterworth filters" in supplement
    assert "the three unsupported boundary rows are exactly zero" in supplement
    assert "unsupported boundary rows are padding-masked and excluded from attentive/mean/max pooling" in supplement
    assert "same synchronized 8 s crop" in supplement
    assert "mean of the accepted beat maxima and minima" in supplement
    assert "curated scalar labels paired with the released 8 s record" in supplement
    assert "Primary repeated-measures Bland--Altman agreement" in supplement
    assert "Holm adjustment across the fixed 22-test family" in supplement
    assert "Apparent-PAT interaction across the fixed timing prior" in supplement
    assert "Zero-shot source-training-seed stability" in supplement
    assert r"\label{tab:source_seed_stability}" in supplement
    assert "Performance across fixed reference-BP strata" in supplement
    assert "each model relative to its own unperturbed synchronized baseline" in supplement
    assert "Shifted offsets 9--12 (516--828 ms support)" in supplement
    assert "Shifted 520--850 ms" not in supplement


def test_dbp_evidence_hierarchy_is_coherent_without_rowwise_tuning():
    main = pd.read_csv(ROOT / "artifacts/metrics/main/main_window_metrics.csv")
    mechanism = pd.read_csv(ROOT / "artifacts/metrics/mechanism/mechanism_window_metrics.csv")
    physiocat = float(main[(main.model == "physiocat") & (main.outcome == "DBP")].mae.iloc[0])
    mufubp = float(main[(main.model == "mufubp_net") & (main.outcome == "DBP")].mae.iloc[0])
    ablations = mechanism[(mechanism.model.isin(["without_pretraining", "without_sqi_fusion"])) & (mechanism.outcome == "DBP")].mae.to_numpy(float)
    assert physiocat < ablations.min()
    assert ablations.max() + 0.02 < mufubp


def test_reviewer_readme_states_the_aligned_target_boundary():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    contract = (ROOT / "docs/SCIENTIFIC_CONTRACT.md").read_text(encoding="utf-8")
    assert "offline label-agreement auditing" not in readme + contract
    assert "released scalar-label alignment" not in readme + contract
    assert "Source rows are enumerated before crop and target formation" in contract
    assert "one source-listed segment can yield at most one deterministic centered 8-s crop" in contract
    assert "same released `aligned_abp_scalar_targets`/`abp_beat_labels` path" in contract
    assert "PulseDB scalar targets use ECG-detected cardiac boundaries" in contract
    assert "Reference ABP is never a model or inference input" in readme and contract
    assert "label-source sensitivity analysis" in contract
    assert "assets/physiocat_architecture.png" in readme
    assert "assets/subject_grouped_results.png" in readme


def test_manuscript_matches_edge_aligned_reciprocal_implementation():
    manuscript = (ROOT / "paper/PhysioCAT_Manuscript.tex").read_text(encoding="utf-8")
    assert "combined on the same admissible ECG--PPG edges" in manuscript
    assert "ECG-query/PPG-key" in manuscript and "PPG-query/ECG-key" in manuscript
    assert r"M_{E\leftarrow P}" in manuscript and r"M_{P\leftarrow E}" in manuscript
    assert r"H_E + W_{EP}" not in manuscript and r"H_P + W_{PE}" not in manuscript
    assert r"F_i=\bar r_i" in manuscript
    assert "fit normalization and pre-training statistics" not in manuscript
    assert "This deterministic transform fitted no subject- or cohort-level statistics" in manuscript
    assert "lowered window-weighted MAE" in manuscript
    assert "literature-guided 120--450 ms lag envelope was held fixed throughout model comparison [13,19--20,42--44]" in manuscript
    assert "120--450 ms attention band was held fixed throughout model comparison [45]" not in manuscript
    assert "The mask acts on fixed patch timestamps rather than detected R-wave or pulse-onset events" in manuscript
    assert "apparent PAT is used only in descriptive mechanism analyses" in manuscript
    assert "two separately curated, patient-disjoint target protocols from the same MIMIC-III waveform ecosystem" in manuscript
    assert "target-protocol identity audit found" not in manuscript
    assert "25b910030@stu.hit.edu.cn" in manuscript
    assert "each measured relative to its own synchronized unperturbed baseline" in manuscript
    assert "UMAP: Uniform Manifold Approximation and Projection" not in manuscript
    assert "Supplementary Material [57--59]" in manuscript


def test_release_metadata_uses_current_version():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_init = (ROOT / "src/physiocat/__init__.py").read_text(encoding="utf-8")
    defaults = yaml.safe_load((ROOT / "configs/defaults.yaml").read_text(encoding="utf-8"))
    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation = yaml.safe_load(citation_text)
    manifest = json.loads((ROOT / "REPOSITORY_MANIFEST.json").read_text(encoding="utf-8"))
    assert 'version = "1.0.0"' in pyproject
    assert '__version__ = "1.0.0"' in package_init
    assert str(defaults["version"]) == "1.0.0"
    assert "version: 1.0.0" in citation_text
    assert str(citation["version"]) == "1.0.0"
    assert str(citation["date-released"]) == "2026-07-29"
    assert citation["authors"][0]["affiliation"] == "eHealth Research Institute, School of Management, Harbin Institute of Technology; Faculty of Computing, Harbin Institute of Technology"
    assert citation["authors"][1]["affiliation"] == "eHealth Research Institute, School of Management, Harbin Institute of Technology"
    assert citation["authors"][1]["email"] == "25b910030@stu.hit.edu.cn"
    assert manifest["artifact_version"] == "1.0.0"
    assert manifest["review_snapshot_tag"] == "bspc-submission-v1"


def test_public_author_and_affiliation_metadata_are_consistent():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    manuscript = (ROOT / "paper/PhysioCAT_Manuscript.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "paper/PhysioCAT_Supplementary_Material.tex").read_text(encoding="utf-8")
    assert "Youren Shang · Ningyuan Zhang — eHealth Research Institute, Harbin Institute of Technology" in readme
    assert "(corresponding author)" not in readme
    assert r"\author[inst1,inst2]{Youren Shang}" in manuscript
    assert r"\author[inst1]{Ningyuan Zhang\corref{cor1}}" in manuscript
    assert r"\address[inst1]{eHealth Research Institute, School of Management, Harbin Institute of Technology, Harbin 150001, China}" in manuscript
    assert r"\address[inst2]{Faculty of Computing, Harbin Institute of Technology, Harbin 150001, China}" in manuscript
    assert r"Youren Shang$^{1,2}$, Ningyuan Zhang$^{1,*}$" in supplement
    assert r"$^{2}$Faculty of Computing, Harbin Institute of Technology, Harbin 150001, China" in supplement


def test_data_availability_uses_a_non_self_referential_release_identity():
    manuscript = (ROOT / "paper/PhysioCAT_Manuscript.tex").read_text(encoding="utf-8")
    versioning = (ROOT / "VERSIONING.md").read_text(encoding="utf-8")
    assert "PhysioCAT v1.0.0" in manuscript
    assert r"\texttt{bspc-submission-v1}" in manuscript
    assert r"\url{https://github.com/shangyr/PhysioCAT/releases/tag/bspc-submission-v1}" in manuscript
    assert "The Release records the target Git commit and provides the source archive" in manuscript
    assert "recorded in the submission system" not in manuscript
    assert "manuscript-management system" not in versioning
    assert "outside the tracked source" in versioning
    assert re.search(r"\b[0-9a-f]{40}\b", manuscript) is None


def test_funding_and_non_author_pi_boundaries_are_explicit():
    manuscript = (ROOT / "paper/PhysioCAT_Manuscript.tex").read_text(encoding="utf-8")
    assert "National Natural Science Foundation of China [grant number 72125001]" in manuscript
    assert "The funding source had no role in the study design" in manuscript
    assert "decision to submit the article for publication" in manuscript
    assert "did not receive any specific grant" not in manuscript
    assert "Professor Xitong Guo" in manuscript
    assert manuscript.count("Xitong Guo") == 1
    assert r"\author{Xitong Guo" not in manuscript


def test_released_checkpoint_demo_replays_prediction_authority():
    script = ROOT / "examples/released_checkpoint_demo.py"
    spec = importlib.util.spec_from_file_location("released_checkpoint_demo_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    report = module.run_demo()
    assert report["status"] == "PASS"
    assert report["windows"] == 8
    assert report["output_shape"] == [8, 2]
    assert report["max_abs_replay_delta"] <= 5e-5


def test_readme_matches_the_released_normalization_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    contract = (ROOT / "docs/SCIENTIFIC_CONTRACT.md").read_text(encoding="utf-8")
    assert "same deterministic whole-window robust-normalized ECG/PPG crop" in contract
    assert "Patch-local normalization is retained only as a separately named sensitivity control" in contract
    assert "normalized independently within each non-overlapping 64-ms analysis-grid patch" not in readme + contract


def test_reviewer_guide_routes_existing_claim_boundary_and_hard_case_evidence():
    guide = (ROOT / "REVIEWER_GUIDE.md").read_text(encoding="utf-8")
    required_routes = (
        "Reference-target construction",
        "Fixed-patch delay implementation and mechanism controls",
        "Pressure-range and residual structure",
        "MIMIC target-protocol identity",
        "artifacts/metrics/secondary/conditional_bp_performance.csv",
        "artifacts/cohorts/mimic_ecosystem_identity_audit.csv",
    )
    assert all(route in guide for route in required_routes)
    assert (ROOT / "artifacts/metrics/secondary/conditional_bp_performance.csv").is_file()
    assert (ROOT / "artifacts/metrics/statistics/repeated_measures_agreement.csv").is_file()
    assert (ROOT / "artifacts/cohorts/label_source_audit.csv").is_file()
    assert (ROOT / "artifacts/cohorts/mimic_ecosystem_identity_audit.csv").is_file()


def test_released_profiling_summary_is_derived_from_raw_timing_samples():
    raw = pd.read_csv(ROOT / "artifacts/profiling/runtime_samples.csv.gz")
    environments = pd.read_csv(ROOT / "artifacts/profiling/profile_environments.csv")
    summary = pd.read_csv(ROOT / "artifacts/metrics/secondary/deployment_profile.csv")
    assert raw.groupby("profile_id").size().eq(5000).all()
    observed = raw.groupby("profile_id", as_index=False).agg(
        forward_latency_ms=("forward_latency_ms", "median"),
        end_to_end_latency_ms=("end_to_end_latency_ms", "median"),
    )
    observed = environments[["profile_id", "platform", "runtime"]].merge(observed, on="profile_id")
    merged = summary.merge(observed, on=["profile_id", "platform", "runtime"], suffixes=("_released", "_raw"))
    assert np.allclose(merged.forward_latency_ms_released, merged.forward_latency_ms_raw, atol=5e-6)
    assert np.allclose(merged.end_to_end_latency_ms_released, merged.end_to_end_latency_ms_raw, atol=5e-6)
    assert (ROOT / "scripts/profile/export_onnx.py").is_file()
    assert (ROOT / "scripts/profile/build_tensorrt_engine.py").is_file()


def test_baseline_forward_contracts():
    ecg = torch.randn(1, 1, 2000)
    ppg = torch.randn(1, 1, 2000)
    sqi = torch.rand(1, 2, 125)
    for name in ("cnn_bilstm", "bp_net", "te_sagru", "mufubp_net"):
        model = build_neural_baseline(name).eval()
        with torch.no_grad():
            prediction = model(ecg, ppg, sqi)
        assert prediction.shape == (1, 2)
        assert torch.isfinite(prediction).all()
        assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) > 500_000


def test_fold_role_matrix_is_complete():
    archive = np.load(ROOT / "data/folds/fold_subject_roles.npz", allow_pickle=False)
    roles = archive["roles"]
    assert roles.shape == (5, 2714)
    assert np.all((roles == 2).sum(axis=0) == 1)
    assert np.all((roles == 1).sum(axis=1) == 271)
    assert np.all(((roles == 0) | (roles == 1) | (roles == 2)))
    test = pd.read_csv(ROOT / "data/folds/pulsedb_vital_test_membership.csv.gz")
    validation = pd.read_csv(ROOT / "data/folds/pulsedb_vital_validation_membership.csv.gz")
    subjects = archive["subject_id"].astype(str)
    for fold_id in range(1, 6):
        assert set(test.loc[test.fold_id.eq(fold_id), "test_subject_id"].astype(str)) == set(subjects[roles[fold_id - 1] == 2])
        assert set(validation.loc[validation.fold_id.eq(fold_id), "validation_subject_id"].astype(str)) == set(subjects[roles[fold_id - 1] == 1])


def test_random_segment_protocol_is_complete_five_fold_oof():
    membership = pd.read_csv(
        ROOT / "data/folds/pulsedb_vital_random_segment_5fold_membership.csv.gz"
    )
    summary = pd.read_csv(
        ROOT / "data/folds/pulsedb_vital_random_segment_5fold_summary.csv"
    )
    predictions = pd.read_csv(
        ROOT / "artifacts/predictions/protocol_random_split_predictions.csv.gz"
    )
    primary = pd.read_csv(
        ROOT / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz",
        usecols=["window_id", "subject_id"],
    )
    cohort_registry = pd.read_csv(ROOT / "artifacts/cohorts/cohort_registry.csv").set_index("cohort")
    expected_windows = int(cohort_registry.loc["pulsedb_vital", "windows"])
    assert len(membership) == len(predictions) == len(primary) == expected_windows
    assert membership.window_id.is_unique and predictions.window_id.is_unique
    assert set(membership.window_id) == set(primary.window_id) == set(predictions.window_id)
    assert set(membership.evaluation_fold_id) == set(range(5))
    assert set(membership.prediction_role) == {"out_of_fold_test"}
    assert set(predictions.prediction_role) == {"out_of_fold_test"}
    bound = predictions[["window_id", "evaluation_fold_id"]].merge(
        membership[["window_id", "evaluation_fold_id"]],
        on="window_id",
        suffixes=("_prediction", "_membership"),
        validate="one_to_one",
    )
    assert np.array_equal(bound.evaluation_fold_id_prediction, bound.evaluation_fold_id_membership)
    assert len(summary) == 5 and (summary.prediction_role == "out_of_fold_test").all()
    assert (summary.test_subject_overlap_pct > 99.0).all()
    for fold_id in range(5):
        test = set(membership.loc[membership.segment_partition_id == fold_id, "window_id"])
        validation = set(membership.loc[membership.segment_partition_id == (fold_id + 1) % 5, "window_id"])
        train = set(membership.loc[~membership.segment_partition_id.isin([fold_id, (fold_id + 1) % 5]), "window_id"])
        assert not (train & validation or train & test or validation & test)
        assert len(train | validation | test) == len(membership)


def test_configuration_registry_and_prediction_authority_manifest_are_complete():
    selected = pd.read_csv(ROOT / "artifacts/logs/training/model_configuration_registry.csv")
    manifest = pd.read_csv(ROOT / "artifacts/predictions/prediction_authority_manifest.csv")
    assert len(selected) == 50
    assert selected.configuration_sha256.is_unique
    assert selected.selection_scope.str.contains("fold-local validation", regex=False).all()
    assert int((selected.outer_folds == 5).sum()) == 50
    assert selected.outer_folds.eq(5).all()
    random_masks = selected[selected.model.str.startswith("attention_edge_ablation_seed_")]
    assert len(random_masks) == 20
    assert random_masks.training_kind.eq("neural").all()
    assert random_masks.initialization_seed.eq(42).all()
    assert random_masks.data_order_seed.eq(42).all()
    assert set(random_masks.mask_seed.astype(int)) == set(range(82000, 82020))
    assert all(row.input_view == input_view_for_model(row.model) for row in selected.itertuples(index=False))
    assert len(manifest) == 7 and manifest.artifact.is_unique
    for row in manifest.itertuples(index=False):
        path = ROOT / row.artifact
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.artifact_sha256
        frame = pd.read_csv(path, compression="gzip", low_memory=False)
        prediction_columns = sorted(column for column in frame if column.startswith("pred_"))
        if not prediction_columns:
            prediction_columns = sorted(
                column for column in frame
                if column in {"n_windows", "absolute_error_sum", "signed_error_sum"}
            )
        numeric = frame[prediction_columns].to_numpy(dtype="<f8", copy=True)
        payload = ("\n".join(prediction_columns) + "\n").encode("utf-8") + numeric.tobytes(order="C")
        assert hashlib.sha256(payload).hexdigest() == row.prediction_values_sha256
        assert len(frame) == int(row.rows) and len(prediction_columns) == int(row.prediction_columns)
    matched = selected.set_index("model").loc[["physiocat", "matched_no_delay"]]
    for column in ("learning_rate", "weight_decay", "batch_size", "pretrain_epochs", "maximum_epochs", "early_stopping_patience", "initialization_seed", "data_order_seed", "mask_seed", "checkpoint_rule"):
        assert matched[column].nunique() == 1


def test_design_parameters_and_training_setting_selection_exclude_outer_test_metrics():
    design = pd.read_csv(ROOT / "artifacts/protocol/design_parameter_provenance.csv")
    selection = pd.read_csv(ROOT / "artifacts/logs/training/fold_local_training_setting_selection.csv")
    assert {"patch_samples", "delay_envelope_ms", "implemented_offsets", "sqi_rule", "subject_window_cap"}.issubset(set(design.item))
    assert len(selection) == 6 * 5 * 3
    assert set(selection.outer_fold_id) == {1, 2, 3, 4, 5}
    assert not selection.outer_test_metrics_available_to_selection.astype(bool).any()
    grouped = selection.groupby(["model", "outer_fold_id"])
    assert grouped.size().eq(3).all()
    assert grouped.selected.sum().eq(1).all()
    chosen = selection[selection.selected].set_index(["model", "outer_fold_id"]).validation_mean_component_mae.sort_index()
    minimum = grouped.validation_mean_component_mae.min().sort_index()
    assert np.allclose(chosen, minimum)


def test_every_reported_neural_configuration_routes_through_the_public_training_entrypoint():
    spec = importlib.util.spec_from_file_location("physiocat_train_fold_entrypoint", ROOT / "scripts/train/train_fold.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    registry = pd.read_csv(ROOT / "artifacts/logs/training/model_configuration_registry.csv")
    reported_neural = registry[registry.outer_folds.eq(5) & registry.training_kind.eq("neural")]
    for name in reported_neural.model:
        model = module.model_for(str(name))
        assert isinstance(model, torch.nn.Module), name
        del model


def test_attention_export_is_bound_to_subject_grouped_outer_test_rows():
    manifest = pd.read_csv(ROOT / "artifacts/attention/attention_export_manifest.csv").iloc[0]
    summary = pd.read_csv(ROOT / "artifacts/attention/window_level_attention_summary.csv.gz")
    archive = np.load(ROOT / "artifacts/attention/sparse_attention_weights.npz", allow_pickle=False)
    primary = pd.read_csv(
        ROOT / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz",
        usecols=["window_id", "subject_id", "pat_detected", "evaluation_fold_id"],
    )
    eligible = primary[primary.pat_detected.astype(bool)].copy()
    counts = eligible.groupby("subject_id").size()
    pool = pd.Series(counts[counts >= int(manifest.windows_per_subject)].index.astype(str), name="subject_id")
    chosen = pool.sample(n=int(manifest.subjects), random_state=int(manifest.selection_seed)).sort_values()
    subject_values = eligible.subject_id.astype(str)
    parts = []
    for subject in chosen:
        subject_seed = int.from_bytes(
            hashlib.sha256(f"{int(manifest.selection_seed)}|{subject}".encode("utf-8")).digest()[:4], "little"
        )
        parts.append(
            eligible.loc[subject_values == subject].sample(
                n=int(manifest.windows_per_subject), random_state=subject_seed
            )
        )
    expected = pd.concat(parts, ignore_index=True).sort_values(["subject_id", "window_id"]).reset_index(drop=True)
    assert manifest.selection == "fixed-seed balanced four-window-per-subject subject-grouped OOF sample"
    assert np.array_equal(summary.window_id.astype(str), expected.window_id.astype(str))
    assert summary.groupby("subject_id").size().eq(int(manifest.windows_per_subject)).all()
    assert int(manifest.multiwindow_subjects) == int(manifest.subjects)
    assert set(summary.evaluation_role.astype(str)) == {"outer_test"}
    subject_fold = primary[["subject_id", "evaluation_fold_id"]].drop_duplicates()
    assert subject_fold.subject_id.nunique() == len(subject_fold)
    fold_by_subject = dict(zip(subject_fold.subject_id.astype(str), subject_fold.evaluation_fold_id.astype(int), strict=True))
    expected_folds = summary.subject_id.astype(str).map(fold_by_subject).to_numpy(np.int32)
    assert np.array_equal(summary.outer_fold_id.to_numpy(np.int32), expected_folds)
    assert np.array_equal(archive["outer_fold_id"].astype(np.int32), expected_folds)
    assert int(manifest.subjects) == summary.subject_id.nunique()
    assert int(manifest.outer_folds) == summary.outer_fold_id.nunique()



def test_random_mask_prediction_authority_matches_seed82000_control():
    archive = np.load(ROOT / "artifacts/predictions/random_mask_20_seed_predictions.npz", allow_pickle=False)
    mechanism = pd.read_csv(ROOT / "artifacts/predictions/mechanism_control_predictions.csv.gz", compression="gzip")
    assert archive["predicted_sbp"].shape[0] == 20
    assert len(np.unique(archive["subject_id"])) == 2714
    assert str(archive["evaluation_scope"][0]) == "independently trained five-fold random-mask topology replication"
    assert np.array_equal(archive["seed"], np.arange(82000, 82020))
    authority = mechanism.set_index("window_id").loc[archive["window_id"].astype(str)]
    assert np.allclose(archive["predicted_sbp"][0], authority.pred_attention_edge_ablation_sbp, atol=5e-6)
    assert np.allclose(archive["predicted_dbp"][0], authority.pred_attention_edge_ablation_dbp, atol=5e-6)
    summary = pd.read_csv(ROOT / "artifacts/metrics/mechanism/random_mask_20_seed_summary.csv")
    assert summary.evaluation_scope.eq("independently trained five-fold random-mask topology replication").all()
    assert summary.subjects.eq(2714).all() and summary.outer_folds.eq(5).all()
    assert summary.mask_seeds.eq(20).all()
    supplement = (ROOT / "paper/PhysioCAT_Supplementary_Material.tex").read_text(encoding="utf-8")
    assert "five-fold subject-grouped" in supplement
    assert "independently trained" in supplement and "random-rewiring" in supplement


def test_seed42_stability_rows_are_exact_main_prediction_authorities():
    stability = pd.read_csv(ROOT / "artifacts/metrics/stability/three_seed_subject_grouped_summary.csv")
    predictions = pd.read_csv(ROOT / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz", low_memory=False)
    mechanism = pd.read_csv(ROOT / "artifacts/predictions/mechanism_control_predictions.csv.gz", low_memory=False)
    seed42 = stability[stability.seed == 42]
    assert len(seed42) == 5 * 4 * 2
    authorities = []
    subject_fold = predictions[["subject_id", "evaluation_fold_id"]].drop_duplicates().set_index("subject_id").evaluation_fold_id
    for model in sorted(seed42.model.unique()):
        authority_frame = mechanism if model == "ppg_leading_mirror" else predictions
        for outcome in sorted(seed42.outcome.unique()):
            target = outcome.lower()
            subject_mae = (
                (authority_frame[f"pred_{model}_{target}"] - authority_frame[target])
                .abs()
                .groupby(authority_frame.subject_id, sort=False)
                .mean()
            )
            fold_mae = subject_mae.groupby(subject_fold.loc[subject_mae.index]).mean().rename("authority_fold_mae").reset_index().rename(columns={"evaluation_fold_id": "fold_id"})
            fold_mae["model"] = model
            fold_mae["outcome"] = outcome
            authorities.append(fold_mae)
    authority = pd.concat(authorities, ignore_index=True)
    checked = seed42.merge(
        authority,
        on=["fold_id", "model", "outcome"],
        how="left",
        validate="one_to_one",
    )
    assert checked.authority_fold_mae.notna().all()
    assert np.allclose(checked.authority_fold_mae, checked.fold_mae, atol=5e-6, rtol=0.0)


def test_replay_logs_do_not_masquerade_as_final_training():
    manifest = pd.read_csv(ROOT / "artifacts/replay/replay_manifest.csv")
    for row in manifest.itertuples(index=False):
        log = pd.read_csv(ROOT / row.training_log)
        expected_pretraining = 5 if row.model in {"physiocat", "matched_no_delay"} else 0
        assert int((log.stage == "contrastive").sum()) == expected_pretraining
        assert int((log.stage == "supervised").sum()) == 20
        assert log.artifact_scope.str.contains("not final subject-grouped outer-fold training", regex=False).all()


def test_source_model_protocol_predictions_and_external_manifest_are_complete():
    protocol = pd.read_csv(ROOT / "artifacts/protocol/source_model_protocol.csv")
    assert set(protocol.source_model) == {"physiocat", "matched_no_delay", "mufubp_net"}
    assert (protocol.training_subjects == 2442).all() and (protocol.validation_subjects == 272).all()
    metrics = pd.read_csv(ROOT / "artifacts/metrics/external/source_model_internal_validation.csv")
    for row in protocol.itertuples(index=False):
        assert row.input_view == input_view_for_model(row.source_model)
        predictions = pd.read_csv(ROOT / row.validation_predictions, compression="gzip")
        assert len(predictions) == int(row.validation_windows)
        assert predictions.subject_id.nunique() == int(row.validation_subjects)
        assert hashlib.sha256((ROOT / row.validation_predictions).read_bytes()).hexdigest() == row.validation_predictions_sha256
        for outcome in ("SBP", "DBP"):
            target = outcome.lower()
            observed = float(np.mean(np.abs(predictions[f"predicted_{target}"] - predictions[f"reference_{target}"])))
            expected = float(metrics[(metrics.source_model == row.source_model) & (metrics.outcome == outcome)].mae.iloc[0])
            assert abs(observed - expected) < 7e-7
    manifest = pd.read_csv(ROOT / "artifacts/protocol/external_prediction_manifest.csv")
    assert len(manifest) == 6
    assert set(manifest.target_cohort) == {"pulsedb_mimic", "mimic_bp"}
    expected = protocol.set_index("source_model").configuration_sha256.to_dict()
    assert manifest.apply(lambda row: row.source_configuration_sha256 == expected[row.source_model], axis=1).all()
    assert manifest.apply(lambda row: row.source_input_view == input_view_for_model(row.source_model), axis=1).all()
    assert not manifest.target_tuning.astype(bool).any()


def test_source_training_seed_stability_is_complete_and_non_degenerate():
    metrics = pd.read_csv(ROOT / "artifacts/metrics/external/source_model_three_seed_metrics.csv")
    summary = pd.read_csv(ROOT / "artifacts/metrics/external/source_model_three_seed_summary.csv")
    protocol = pd.read_csv(ROOT / "artifacts/protocol/source_model_three_seed_protocol.csv")
    statistics = pd.read_csv(
        ROOT / "artifacts/predictions/source_model_three_seed_subject_statistics.csv.gz"
    )
    assert len(protocol) == 9
    assert set(protocol.source_seed) == {42, 1337, 2025}
    assert set(protocol.source_model) == {"physiocat", "matched_no_delay", "mufubp_net"}
    assert not protocol.target_tuning.astype(bool).any()
    assert set(metrics.evaluation_scope) == {"pulsedb_vital_source_validation", "pulsedb_mimic", "mimic_bp"}
    assert len(metrics) == 3 * 3 * 3 * 2
    assert summary.source_seeds.eq(3).all()
    assert (summary.window_mae_sd > 0.005).all()
    assert (summary.window_mae_sd < 0.20).all()
    for scope in ("pulsedb_mimic", "mimic_bp"):
        subset = metrics[metrics.evaluation_scope.eq(scope)]
        for seed in (42, 1337, 2025):
            rows = subset[subset.source_seed.eq(seed)]
            for outcome in ("SBP", "DBP"):
                indexed = rows[rows.outcome.eq(outcome)].set_index("model").window_mae
                assert indexed["physiocat"] < indexed["matched_no_delay"]
    assert set(statistics.source_seed.astype(int)) == {42, 1337, 2025}
    assert set(statistics.model.astype(str)) == {"physiocat", "matched_no_delay", "mufubp_net"}
    assert set(statistics.evaluation_scope) == {"pulsedb_vital_source_validation", "pulsedb_mimic", "mimic_bp"}
    assert set(statistics.outcome) == {"SBP", "DBP"}
    assert (statistics.n_windows > 0).all() and (statistics.absolute_error_sum >= 0).all()


def test_pat_stratified_mechanism_interaction_is_complete_and_natural():
    table = pd.read_csv(ROOT / "artifacts/metrics/secondary/pat_stratified_model_comparison.csv")
    contrasts = pd.read_csv(ROOT / "artifacts/metrics/secondary/pat_group_interaction_contrasts.csv")
    primary = table[table.group_type.eq("primary")]
    assert set(primary.pat_group) == {
        "PAT detected: 120--450 ms", "PAT detected: outside 120--450 ms", "PAT not detected"
    }
    assert set(primary.model) == {"physiocat", "matched_no_delay", "ppg_leading_mirror", "direction_agnostic_local"}
    assert set(primary.outcome) == {"SBP", "DBP"}
    assert primary.groupby(["pat_group", "model"]).outcome.nunique().eq(2).all()
    nod = primary[primary.model.eq("matched_no_delay")].set_index(["pat_group", "outcome"])
    in_sbp = float(nod.loc[("PAT detected: 120--450 ms", "SBP"), "mae_reduction_vs_physiocat"])
    out_sbp = float(nod.loc[("PAT detected: outside 120--450 ms", "SBP"), "mae_reduction_vs_physiocat"])
    missing_sbp = float(nod.loc[("PAT not detected", "SBP"), "mae_reduction_vs_physiocat"])
    assert 0.65 < in_sbp < 1.25
    assert 0.15 < out_sbp < in_sbp - 0.10
    assert 0.10 < missing_sbp < in_sbp - 0.10
    nod_contrasts = contrasts[contrasts.comparator_model.eq("matched_no_delay")]
    assert len(nod_contrasts) == 4
    assert (nod_contrasts.difference_in_mae_reduction_mmHg > 0.10).all()
    assert (nod_contrasts.cluster_bootstrap_ci_low > 0).all()


def test_waveform_and_sqi_contexts_are_not_conflated():
    for name in ("physiocat", "matched_no_delay"):
        config = yaml.safe_load((ROOT / f"configs/model/{name}.yaml").read_text(encoding="utf-8"))
        architecture = config["architecture"]
        assert architecture["waveform_encoder_context_ms"] == 64
        assert architecture["sqi_reliability_context_ms"] == 1000
        assert "maximum_encoder_context_ms" not in architecture
    contract = (ROOT / "docs/SCIENTIFIC_CONTRACT.md").read_text(encoding="utf-8")
    assert "64-ms" in contract and "1-s local analysis window" in contract


def test_frozen_source_checkpoint_to_prediction_lineage_is_closed():
    protocol = pd.read_csv(ROOT / "artifacts/protocol/source_model_protocol.csv")
    checkpoints = pd.read_csv(ROOT / "artifacts/checkpoints/checkpoint_manifest.csv", keep_default_na=False)
    checkpoints = checkpoints[checkpoints.checkpoint_role == "frozen_source_model"].set_index("model")
    lineage = pd.read_csv(ROOT / "artifacts/protocol/source_model_prediction_lineage.csv")
    external = pd.read_csv(ROOT / "artifacts/protocol/external_prediction_manifest.csv")
    assert len(lineage) == 9
    assert (lineage.groupby("source_model").size() == 3).all()
    assert set(lineage.prediction_role) == {"source_validation", "zero_shot_external"}
    for row in protocol.itertuples(index=False):
        checkpoint = checkpoints.loc[row.source_model]
        predictions = pd.read_csv(ROOT / row.validation_predictions)
        sbp_mae = float(np.mean(np.abs(predictions.predicted_sbp - predictions.reference_sbp)))
        dbp_mae = float(np.mean(np.abs(predictions.predicted_dbp - predictions.reference_dbp)))
        selected = pd.read_csv(ROOT / checkpoint.training_log)
        selected = selected[selected.selected_checkpoint.astype(bool)].iloc[0]
        assert row.checkpoint == checkpoint.checkpoint
        assert row.checkpoint_sha256 == checkpoint.checkpoint_sha256
        assert row.checkpoint_state_dict_sha256 == checkpoint.state_dict_sha256
        assert "declared source" in row.release_scope
        assert "no checkpoint is represented" not in row.release_scope
        assert abs(selected.validation_sbp_mae - sbp_mae) < 1e-7
        assert abs(selected.validation_dbp_mae - dbp_mae) < 1e-7
        bound_external = external[external.source_model == row.source_model]
        assert len(bound_external) == 2
        assert set(bound_external.source_checkpoint_sha256) == {checkpoint.checkpoint_sha256}
        assert set(bound_external.source_checkpoint_state_dict_sha256) == {checkpoint.state_dict_sha256}
        for lineage_row in lineage[lineage.source_model == row.source_model].itertuples(index=False):
            prediction_path = ROOT / lineage_row.prediction_artifact
            assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == lineage_row.prediction_artifact_sha256
            assert hashlib.sha256((ROOT / lineage_row.input_manifest).read_bytes()).hexdigest() == lineage_row.input_manifest_sha256
            assert hashlib.sha256((ROOT / lineage_row.preprocessing_config).read_bytes()).hexdigest() == lineage_row.preprocessing_config_sha256


def test_all_reported_p_values_are_finite_and_positive():
    table = pd.read_csv(ROOT / "artifacts/metrics/statistics/paired_subject_tests.csv")
    assert len(table) == 22
    for column in ("raw_p", "holm_adjusted_p"):
        values = table[column].to_numpy(float)
        assert np.isfinite(values).all()
        assert (values > 0).all()
    expected = np.maximum(holm_adjust(table.raw_p.to_numpy(float)), np.finfo(float).tiny)
    assert np.allclose(table.holm_adjusted_p.to_numpy(float), expected, atol=1e-12, rtol=1e-12)
    assert set(table.comparator_model) >= {
        "uniform_delay_band", "ppg_leading_mirror", "direction_agnostic_local", "shifted_offsets_9_12", "attention_edge_ablation"
    }


def test_rr_irregularity_and_external_patient_identity_are_recomputable():
    audit = pd.read_csv(ROOT / "data/retention/pulsedb_vital_window_audit.csv.gz")
    expected = np.sqrt(audit.rr_cv.to_numpy(float) ** 2 + audit.rr_rmssd_ratio.to_numpy(float) ** 2)
    assert np.allclose(audit.rr_irregularity.to_numpy(float), expected, atol=5e-7, rtol=0)
    identity = pd.read_csv(ROOT / "artifacts/cohorts/mimic_ecosystem_identity_audit.csv").iloc[0]
    assert int(identity.shared_source_patient_hashes) == 0
    assert int(identity.shared_source_record_hashes) == 0


def test_source_shift_projection_round_trip():
    archive = np.load(ROOT / "artifacts/representations/source_shift_input_features.npz", allow_pickle=False)
    raw = archive["input_summary_features"].astype(float)
    standardized = archive["standardized_features"].astype(float)
    mean = archive["feature_mean"].astype(float)
    sd = archive["feature_sd"].astype(float)
    loadings = archive["pca_loadings"].astype(float)
    coordinates = archive["pca_coordinates"].astype(float)
    assert raw.shape[1] == len(archive["feature_names"])
    assert np.allclose((raw - mean) / sd, standardized, atol=2e-5)
    assert np.allclose(standardized @ loadings, coordinates, atol=5e-5)


def test_representative_waveform_rows_are_exactly_bound_to_figure_metadata():
    rows = pd.read_csv(ROOT / "artifacts/metrics/secondary/representative_windows.csv")
    assert set(rows.role) == {"easier", "harder"}
    assert rows.waveform_asset.nunique() == 1 and rows.waveform_sha256.nunique() == 1
    asset = ROOT / rows.waveform_asset.iloc[0]
    import hashlib
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == rows.waveform_sha256.iloc[0]
    archive = np.load(asset, allow_pickle=False)
    for row in rows.itertuples(index=False):
        index = int(row.waveform_row)
        assert str(archive["role"][index]) == row.role
        assert str(archive["window_id"][index]) == row.window_id
        assert str(archive["subject_id"][index]) == row.subject_id
        assert archive["ecg"][index].shape == archive["ppg"][index].shape == (2000,)


def test_structured_mask_controls_match_query_support_and_declared_density():
    assert active_query_rows().all()
    for reverse in (False, True):
        reference = mechanism_mask("delay_asymmetric", reverse=reverse)
        assert mask_row_audit(reference) == {
            "total_edges": 482, "active_rows": 122, "empty_rows": 3,
            "unique_row_counts": [0, 1, 2, 3, 4],
        }
        random = mechanism_mask("random", reverse=reverse, seed=82000)
        assert torch.equal(random.sum(1), reference.sum(1))
        assert int(random.sum()) == 482
    local_forward = mechanism_mask("local")
    local_reverse = mechanism_mask("local", reverse=True)
    assert torch.equal(local_forward, local_reverse.transpose(0, 1))
    assert mask_row_audit(local_forward) == {
        "total_edges": 494, "active_rows": 125, "empty_rows": 0,
        "unique_row_counts": [2, 3, 4],
    }
    local_row, local_col = torch.where(local_forward)
    local_offsets = local_col - local_row
    assert torch.equal(torch.unique(local_offsets), torch.tensor([-2, -1, 1, 2]))
    assert int((local_offsets < 0).sum()) == int((local_offsets > 0).sum()) == 247
    assert int((local_offsets == 0).sum()) == 0
    shifted_forward = mechanism_mask("shifted")
    shifted_reverse = mechanism_mask("shifted", reverse=True)
    assert torch.equal(shifted_forward, shifted_reverse.transpose(0, 1))
    assert mask_row_audit(shifted_forward) == {
        "total_edges": 452, "active_rows": 113, "empty_rows": 12,
        "unique_row_counts": [0, 4],
    }
    assert mask_row_audit(shifted_reverse) == {
        "total_edges": 452, "active_rows": 116, "empty_rows": 9,
        "unique_row_counts": [0, 1, 2, 3, 4],
    }
    for kind in ("delay_asymmetric", "shifted", "local", "random"):
        for reverse in (False, True):
            mask = mechanism_mask(kind, reverse=reverse, seed=82000)
            assert mask.shape == (125, 125)
    random_forward = mechanism_mask("random", seed=82000)
    random_reverse = mechanism_mask("random", reverse=True, seed=82000)
    reference_forward = mechanism_mask("delay_asymmetric")
    assert torch.equal(random_reverse, random_forward.transpose(0, 1))
    assert torch.equal(random_forward.sum(0), reference_forward.sum(0))
    assert torch.equal(random_forward.sum(1), reference_forward.sum(1))
    assert not torch.equal(random_forward, reference_forward)
    assert torch.equal(random_forward, mechanism_mask("random", seed=82000))
    assert not torch.equal(random_forward, mechanism_mask("random", seed=82001))
    for reverse in (False, True):
        mirror = mechanism_mask("mirror", reverse=reverse)
        reference = mechanism_mask("delay_asymmetric", reverse=not reverse)
        assert torch.equal(mirror, reference)
        assert int(mirror.sum()) == 482
    no_delay = mechanism_mask("no_delay")
    assert set(no_delay.sum(1).tolist()) == {0, 125}
    assert int((no_delay.sum(1) > 0).sum()) == 122
    assert int(no_delay.sum()) == 122 * 125
    audit = pd.read_csv(ROOT / "artifacts/metrics/mechanism/mask_row_offset_audit.csv")
    interior = audit[(audit.mask_kind == "mirror") & (audit.edge_count == 4)]
    assert len(interior) > 0 and interior.complete_default_absolute_lag_support.astype(bool).all()
    local_audit = audit[audit.mask_kind == "local"]
    assert local_audit.zero_edges.sum() == 0
    assert local_audit.negative_edges.sum() == local_audit.positive_edges.sum()
    assert local_audit.reciprocal_transpose_match.astype(bool).all()


def test_subject_aware_loss_excludes_same_subject_negatives():
    a = torch.eye(4); b = torch.eye(4)
    subjects = torch.tensor([1, 1, 2, 3])
    loss = subject_aware_nt_xent(a, b, subjects)
    baseline = subject_aware_nt_xent(a, b, torch.arange(4))
    assert torch.isfinite(loss)
    assert loss < baseline


def test_paired_augmentation_preserves_relative_temporal_shift():
    ecg = torch.zeros(2, 1, 64); ppg = torch.zeros_like(ecg)
    ecg[:, :, 20] = 1.0; ppg[:, :, 30] = 1.0
    e, p = paired_augment(ecg, ppg)
    assert torch.argmax(e[0, 0]) - torch.argmax(p[0, 0]) == -10
    assert torch.argmax(e[1, 0]) - torch.argmax(p[1, 0]) == -10


def test_cosine_warmup_is_monotone_then_decays():
    values = [_cosine_warmup(i, 20, 5) for i in range(20)]
    assert values[0] < values[4]
    assert values[5] <= values[4]
    assert values[-1] < values[5]


def test_sqi_golden_and_token_hop_contract():
    ecg = np.sin(np.linspace(0, 20 * np.pi, 2000))
    ppg = np.sin(np.linspace(0, 12 * np.pi, 2000)) + 2.0
    peaks = np.arange(100, 1900, 250)
    assert 0 <= ecg_sqi(ecg, peaks) <= 1
    assert 0 <= ppg_sqi(ppg) <= 1
    assert token_sqi(ppg, "ppg").shape == (125,)
    assert "paired_sample_continuity" in __import__("physiocat.preprocessing", fromlist=["REQUIRED_RETENTION_FIELDS"]).REQUIRED_RETENTION_FIELDS


def test_edge_sqi_is_pairwise_and_monotone_by_construction():
    model = PhysioCAT(tiny_config()).eval()
    ecg = torch.randn(1, 1, 2000)
    ppg = torch.randn(1, 1, 2000)
    high = torch.full((1, 2, 125), 0.8)
    low_ecg = high.clone(); low_ecg[:, 0] = 0.2
    with torch.no_grad():
        _, high_attention = model(ecg, ppg, high, return_attention=True)
        _, low_attention = model(ecg, ppg, low_ecg, return_attention=True)
    assert torch.all(low_attention["edge_pair_reliability"] <= high_attention["edge_pair_reliability"] + 1e-7)
    assert torch.all(low_attention["edge_anchor_reliability"] <= high_attention["edge_anchor_reliability"] + 1e-7)


def test_ppg_sqi_is_scale_and_offset_invariant():
    t = np.arange(250) / 250.0
    ppg = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 2.4 * t)
    assert abs(ppg_sqi(ppg) - ppg_sqi(7.3 * ppg + 18.0)) < 1e-10


def test_sample_continuity_synchrony_contract_does_not_read_waveform_delay():
    ecg = np.zeros(2000, dtype=float)
    ppg = np.roll(np.linspace(-1, 1, 2000), 700)
    stable, jitter, diagnostics = synchronization_stability(ecg, ppg, 250)
    assert stable
    assert jitter == 0.0
    assert diagnostics.size == 0
    broken = ppg.copy(); broken[100] = np.nan
    stable, jitter, diagnostics = synchronization_stability(ecg, broken, 250)
    assert not stable and not np.isfinite(jitter) and diagnostics.size == 0


def test_probabilistic_auxiliary_contract():
    ecg = torch.randn(2, 1, 2000); ppg = torch.randn(2, 1, 2000); sqi = torch.rand(2, 2, 125)
    from physiocat.baselines import build_neural_baseline
    mu = build_neural_baseline("mufubp_net").train()
    pred, aux = mu.forward_with_aux(ecg, ppg)
    assert pred.shape == (2, 2) and torch.isfinite(gaussian_kl(aux["mu"], aux["logvar"]))


def test_classical_baseline_parameter_contracts():
    t = np.arange(2000) / 250.0
    ecg = np.stack([np.sin(2 * np.pi * 1.2 * t), np.sin(2 * np.pi * 1.0 * t)])
    ppg = np.stack([np.sin(2 * np.pi * 1.2 * (t - 0.24)), np.sin(2 * np.pi * 1.0 * (t - 0.28))])
    features = pat_ridge_features(ecg, ppg)
    assert features.shape == (2, 3)
    ridge = PATRidge().fit(features, np.asarray([[120.0, 70.0], [130.0, 75.0]]))
    assert ridge.coef_.shape == (4, 2) and ridge.coef_.size == 8
    assert EngineeredRandomForest().n_estimators == 500


def test_released_random_forest_predictions_respect_each_fold_training_label_hull():
    frame = pd.read_csv(ROOT / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz")
    archive = np.load(ROOT / "data/folds/fold_subject_roles.npz", allow_pickle=False)
    subjects, roles = archive["subject_id"].astype(str), archive["roles"]
    for fold_id in range(1, 6):
        train_subjects = set(subjects[roles[fold_id - 1] == 0])
        train = frame[frame.subject_id.astype(str).isin(train_subjects)]
        test = frame[frame.evaluation_fold_id.eq(fold_id)]
        for outcome in ("sbp", "dbp"):
            lower, upper = float(train[outcome].min()), float(train[outcome].max())
            values = test[f"pred_random_forest_{outcome}"].to_numpy(float)
            assert (values >= lower).all() and (values <= upper).all()
            assert not np.isclose(values, lower, atol=1e-7, rtol=0).any()
            assert not np.isclose(values, upper, atol=1e-7, rtol=0).any()


def test_candidate_failure_codes_match_the_released_abp_path():
    cases = {
        "Too few valid ABP beats": "too_few_candidate_abp_beats",
        "Too few finite ordered ABP beats": "too_few_finite_ordered_abp_beats",
        "Too few ABP beats after robust within-window rejection": "too_few_abp_beats_after_robust_rejection",
        "Aligned ECG and ABP crops must be finite": "nonfinite_aligned_ecg_or_abp_crop",
        "Aligned ECG and ABP crops must be one-dimensional and equal length": "aligned_ecg_abp_length_mismatch",
    }
    for message, expected in cases.items():
        assert _candidate_failure_code(
            ValueError(message), paired_crop_pass=True, target_policy="aligned_abp_crop"
        ) == expected


def test_raw_dataset_adapter_emits_complete_lineage():
    (ROOT / "reports").mkdir(exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="adapter_test_", dir=ROOT / "reports")
    tmp_path = Path(temporary.name)
    t = np.arange(1250, dtype=np.float32) / 125.0
    ecg = np.stack([qrs_test_fixture(t, 1.1), qrs_test_fixture(t, 1.2)]).astype(np.float32)
    ppg = np.stack([np.sin(2 * np.pi * 1.1 * (t - 0.24)), np.sin(2 * np.pi * 1.2 * (t - 0.28))]).astype(np.float32)
    abp = np.stack([95.0 + 25.0 * np.sin(2 * np.pi * 1.1 * (t - 0.12)), 97.0 + 25.0 * np.sin(2 * np.pi * 1.2 * (t - 0.14))]).astype(np.float32)
    np.savez(
        tmp_path / "fixture.npz",
        ECG=ecg,
        PPG=ppg,
        ABP=abp,
        Subject=np.asarray(["MIMIC-S000001", "MIMIC-S000002"]),
        Record=np.asarray(["wave-0001", "wave-0002"]),
        Age=np.asarray([61.0, 64.0]),
        Sex=np.asarray(["Female", "Male"]),
    )
    config = tmp_path / "adapter.yaml"
    config.write_text(
        "file_glob: '*.npz'\nsource_sample_rate_hz: 125\ntarget_sample_rate_hz: 250\nwindow_seconds: 8\ntarget_policy: aligned_abp_crop\n"
        "fields:\n  ecg: [ECG]\n  ppg: [PPG]\n  abp: [ABP]\n  sbp: [SBP]\n  dbp: [DBP]\n"
        "  subject_id: [Subject]\n  record_id: [Record]\n  age_years: [Age]\n  sex: [Sex]\n",
        encoding="utf-8",
    )
    output = tmp_path / "prepared.h5"
    lineage = export_dataset(tmp_path, output, config, "fixture")
    assert lineage["windows"] == 2 and lineage["subjects"] == 2
    assert output.exists() and output.with_suffix(".manifest.csv").exists() and output.with_suffix(".candidate_audit.csv").exists() and output.with_suffix(".lineage.json").exists()
    candidate_audit = pd.read_csv(output.with_suffix(".candidate_audit.csv"))
    assert len(candidate_audit) == 2 and set(candidate_audit.status) == {"accepted"}
    records = list(iter_hdf5_records(output, sample_rate_hz=250))
    assert len(records) == 2 and records[0].window_id == "fixture:00000001"
    assert records[0].ecg.shape == (2000,) and records[0].age_years == 61.0 and records[0].sex == "Female"
    assert abs(records[0].sbp - 120.0) < 0.15 and abs(records[0].dbp - 70.0) < 0.15
    assert records[0].label_source == "aligned_8s_abp_beat_extrema"
    assert records[0].abp is not None and records[0].abp.shape == (2000,)
    temporary.cleanup()


def test_pulsedb_adapter_uses_phase_preserving_raw_ecg_and_ppg_fields():
    import yaml
    config = yaml.safe_load((ROOT / "configs/data/pulsedb_adapter.yaml").read_text(encoding="utf-8"))
    assert config["fields"]["ecg"][:4] == ["Subj_Wins/ECG_Record", "ECG_Record", "Subj_Wins/ECG_Raw", "ECG_Raw"]
    assert "ECG_F" not in config["fields"]["ecg"]
    assert config["fields"]["ppg"][:4] == ["Subj_Wins/PPG_Record", "PPG_Record", "Subj_Wins/PPG_Raw", "PPG_Raw"]
    assert "PPG_F" not in config["fields"]["ppg"]
    assert config["target_policy"] == "aligned_abp_crop"
    mimic = yaml.safe_load((ROOT / "configs/data/mimic_bp_adapter.yaml").read_text(encoding="utf-8"))
    assert mimic["target_policy"] == "curated_scalar_same_record"


def test_pulsedb_classic_mat_subj_wins_fields_and_rejection_audit():
    from scipy import io as scipy_io

    (ROOT / "reports").mkdir(exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="pulsedb_mat_", dir=ROOT / "reports")
    tmp_path = Path(temporary.name)
    t = np.arange(1250, dtype=np.float32) / 125.0
    ecg = np.stack([qrs_test_fixture(t, rate) for rate in (1.0, 1.1, 1.2, 1.05)]).astype(np.float32)
    ppg = np.stack([np.sin(2 * np.pi * rate * (t - 0.24)) for rate in (1.0, 1.1, 1.2, 1.05)]).astype(np.float32)
    abp = np.stack(
        [
            95.0 + 25.0 * np.sin(2 * np.pi * 1.0 * (t - 0.12)),
            96.0 + 24.0 * np.sin(2 * np.pi * 1.1 * (t - 0.13)),
            np.zeros_like(t),
            190.0 + 55.0 * np.sin(2 * np.pi * 1.05 * (t - 0.12)),
        ]
    ).astype(np.float32)
    scipy_io.savemat(
        tmp_path / "PulseDB_fixture.mat",
        {
            "Subj_Wins": {
                "ECG_Record": ecg,
                "PPG_Record": ppg,
                "ABP_Record": abp,
                "Subject": np.asarray(["P001", "P002", "P003", "P004"], dtype=object),
                "Record": np.asarray(["R001", "R002", "R003", "R004"], dtype=object),
                "Age": np.asarray([60.0, 61.0, 62.0, 63.0]),
                "Gender": np.asarray(["Female", "Male", "Female", "Male"], dtype=object),
            }
        },
    )
    config = tmp_path / "adapter.yaml"
    config.write_text((ROOT / "configs/data/pulsedb_adapter.yaml").read_text(encoding="utf-8").replace('"**/*.mat"', "'*.mat'"), encoding="utf-8")
    output = tmp_path / "prepared.h5"
    lineage = export_dataset(tmp_path, output, config, "pulsedb_fixture")
    assert lineage["source_rows_enumerated"] == 4
    assert lineage["windows"] == 3 and lineage["rejected_source_rows"] == 1
    manifest = pd.read_csv(output.with_suffix(".manifest.csv"))
    audit = pd.read_csv(output.with_suffix(".candidate_audit.csv"))
    assert set(manifest.ecg_field) == {"Subj_Wins/ECG_Record"}
    assert set(manifest.ppg_field) == {"Subj_Wins/PPG_Record"}
    assert set(audit.status) == {"accepted", "rejected"}
    assert audit.source_row.nunique() == 4 and audit.window_id.replace("", np.nan).notna().sum() == 3
    assert manifest.sbp.max() > 240 and manifest.dbp.max() > 130
    canonical = canonicalize_candidate_audit(
        audit,
        dataset_name="pulsedb_fixture",
        identity_namespace="pulsedb",
        hash_key=b"unit-test-key",
    )
    assert list(canonical.columns) == OUTPUT_COLUMNS
    assert len(canonical) == 4 and canonical.source_record_hash.is_unique
    assert canonical.target_derivation_pass.sum() == 3
    temporary.cleanup()


def test_canonical_identity_hashes_share_the_source_ecosystem_namespace():
    audit = pd.DataFrame(
        {
            "candidate_id": ["candidate-1"],
            "subject_id": ["MIMIC-SUBJECT-1001"],
            "record_id": ["MIMIC-RECORD-2001"],
            "source_file_sha256": ["a" * 64],
            "source_row": [0],
            "ecg_field": ["ECG"],
            "ppg_field": ["PPG"],
            "abp_field": ["ABP"],
            "crop_start_resampled": [100],
            "crop_stop_resampled": [2100],
            "paired_waveform_crop_pass": [True],
            "target_derivation_pass": [True],
            "failure_reason": [""],
            "status": ["accepted"],
            "window_id": ["window-1"],
        }
    )
    key = b"shared-custodian-key"
    pulsedb_view = canonicalize_candidate_audit(
        audit,
        dataset_name="pulsedb_mimic",
        identity_namespace="mimic-iii",
        hash_key=key,
    ).iloc[0]
    mimic_bp_view = canonicalize_candidate_audit(
        audit,
        dataset_name="mimic_bp",
        identity_namespace="mimic-iii",
        hash_key=key,
    ).iloc[0]
    different_patient = audit.copy()
    different_patient.loc[0, "subject_id"] = "MIMIC-SUBJECT-1002"
    other_view = canonicalize_candidate_audit(
        different_patient,
        dataset_name="mimic_bp",
        identity_namespace="mimic-iii",
        hash_key=key,
    ).iloc[0]

    assert pulsedb_view.cohort != mimic_bp_view.cohort
    assert pulsedb_view.source_patient_hash == mimic_bp_view.source_patient_hash
    assert pulsedb_view.source_record_hash == mimic_bp_view.source_record_hash
    assert pulsedb_view.source_patient_hash != other_view.source_patient_hash


def test_complete_five_fold_training_evidence_and_checkpoint_release_contract():
    registry = pd.read_csv(ROOT / "artifacts/logs/training/model_configuration_registry.csv")
    trained = registry[registry.outer_folds.eq(5)]
    random_masks = registry[registry.model.str.startswith("attention_edge_ablation_seed_")]
    authorities = pd.read_csv(ROOT / "artifacts/logs/training/prediction_authority_manifest.csv")
    checkpoints = pd.read_csv(ROOT / "artifacts/checkpoints/checkpoint_manifest.csv")
    attention = pd.read_csv(ROOT / "artifacts/attention/window_level_attention_summary.csv.gz")
    summary = json.loads((ROOT / "artifacts/logs/training/released_training_evidence_summary.json").read_text(encoding="utf-8"))
    formal_ledger = pd.read_csv(ROOT / "artifacts/logs/training/formal_training_run_ledger.csv.gz", keep_default_na=False)
    consistency = json.loads((ROOT / "artifacts/logs/training/core_validation_test_consistency.json").read_text(encoding="utf-8"))
    validation_test_summary = pd.read_csv(ROOT / "artifacts/logs/training/configuration_validation_test_summary.csv")
    stability_ledger = pd.read_csv(ROOT / "artifacts/logs/training/stability_run_ledger.csv.gz")
    source_ledger = pd.read_csv(ROOT / "artifacts/logs/training/source_model_training_run_ledger.csv.gz")
    representativeness = pd.read_csv(ROOT / "data/folds/fold_subset_representativeness.csv")
    assert len(trained) == 50 and len(random_masks) == 20
    assert registry.outer_folds.eq(5).all()
    assert len(authorities) == len(trained) == 50
    assert set(authorities.configuration_id) == set(trained.configuration_id)
    assert len(formal_ledger) == 50 * 5
    assert len(stability_ledger) == 4 * 2 * 5
    assert set(stability_ledger.model) == {"physiocat", "matched_no_delay", "mufubp_net", "ppg_leading_mirror"}
    assert len(source_ledger) == 3 * 3
    assert set(source_ledger.model) == {"physiocat", "matched_no_delay", "mufubp_net"}
    assert set(source_ledger.initialization_seed) == {42, 1337, 2025}
    assert int(source_ledger.initialization_seed.ne(42).sum()) == 6
    assert formal_ledger.groupby("configuration_id").outer_fold_id.nunique().eq(5).all()
    assert stability_ledger.groupby(["model", "initialization_seed"]).outer_fold_id.nunique().eq(5).all()
    random_mask_runs = formal_ledger[formal_ledger.model.str.startswith("attention_edge_ablation_seed_")]
    assert len(random_mask_runs) == 20 * 5
    assert random_mask_runs.groupby("configuration_id").outer_fold_id.nunique().eq(5).all()
    assert random_mask_runs.initialization_seed.eq(42).all()
    assert random_mask_runs.data_order_seed.eq(42).all()
    assert not representativeness.selection_used_model_predictions.astype(bool).any()
    assert len(checkpoints) == 13
    assert int((checkpoints.checkpoint_role == "representative_outer_fold").sum()) == 10
    assert int((checkpoints.checkpoint_role == "frozen_source_model").sum()) == 3
    representative = checkpoints[checkpoints.checkpoint_role == "representative_outer_fold"]
    assert set(representative.outer_fold_id.astype(int)) == {1, 2, 3, 4, 5}
    assert (representative.groupby("outer_fold_id").size() == 2).all()
    assert representative.validation_prediction_subjects.eq(271).all()
    assert all((ROOT / value).is_file() for value in representative.validation_predictions)
    assert all((ROOT / value).is_file() for value in checkpoints.checkpoint)
    assert len(validation_test_summary) == 50
    assert consistency["status"] == "PASS" and consistency["paired_folds"] == 5
    assert consistency["mean_validation_difference_mmHg"] < 0
    assert consistency["mean_test_difference_mmHg"] < 0
    assert "prediction_authority_sha256" in attention
    assert attention.prediction_authority_sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    assert "fold_manifest_sha256" in attention and attention.fold_manifest_sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    assert summary["status"] == "PASS"
    assert summary["five_fold_configurations"] == 50
    assert summary["independently_trained_random_mask_configurations"] == 20
    assert summary["formal_runs"] == 250
    assert summary["additional_stability_runs"] == 40
    assert summary["source_model_runs"] == 9
    assert summary["additional_source_model_seed_runs"] == 6
    assert summary["random_mask_training_runs"] == 100
    assert summary["representative_outer_fold_checkpoints"] == 10
    assert summary["frozen_source_model_checkpoints"] == 3
    assert summary["formal_gpu_hours"] > 0 and summary["stability_gpu_hours"] > 0
    assert "complete five-fold subject-grouped OOF" in summary["scope"]


def test_released_checkpoint_histories_obey_selection_and_patience():
    checkpoints = pd.read_csv(ROOT / "artifacts/checkpoints/checkpoint_manifest.csv", keep_default_na=False)
    for row in checkpoints.itertuples(index=False):
        log = pd.read_csv(ROOT / row.training_log)
        supervised = log[log.stage == "supervised"].copy()
        stopped_epoch = min(int(row.maximum_epochs), int(row.selected_epoch) + int(row.early_stopping_patience))
        assert int(row.stopped_epoch) == stopped_epoch
        assert np.array_equal(supervised.epoch.to_numpy(int), np.arange(1, stopped_epoch + 1))
        selected = supervised[supervised.selected_checkpoint.astype(bool)]
        assert len(selected) == 1 and int(selected.epoch.iloc[0]) == int(row.selected_epoch)
        assert int(supervised.loc[supervised.validation_mean_mae.idxmin(), "epoch"]) == int(row.selected_epoch)
        assert np.allclose(
            selected[["validation_sbp_mae", "validation_dbp_mae", "validation_mean_mae"]].iloc[0].to_numpy(float),
            np.asarray([
                row.selected_validation_sbp_mae,
                row.selected_validation_dbp_mae,
                row.selected_validation_mean_mae,
            ], dtype=float),
            atol=1e-7,
            rtol=0.0,
        )
        expected_stale, expected_best = [], []
        best_value, best_epoch, stale = float("inf"), 0, 0
        for epoch, objective in zip(supervised.epoch, supervised.validation_mean_mae, strict=True):
            if float(objective) < best_value - 1e-12:
                best_value, best_epoch, stale = float(objective), int(epoch), 0
            else:
                stale += 1
            expected_stale.append(stale)
            expected_best.append(best_epoch)
        assert np.array_equal(supervised.epochs_without_improvement.to_numpy(int), np.asarray(expected_stale))
        assert np.array_equal(supervised.best_epoch_so_far.to_numpy(int), np.asarray(expected_best))
        progress = np.maximum(0.0, (supervised.epoch.to_numpy(float) - 5.0) / max(int(row.maximum_epochs) - 5, 1))
        expected_lr = np.where(
            supervised.epoch.to_numpy(int) <= 5,
            float(row.base_learning_rate) * supervised.epoch.to_numpy(float) / 5.0,
            0.5 * float(row.base_learning_rate) * (1.0 + np.cos(np.pi * progress)),
        )
        assert np.allclose(supervised.learning_rate, expected_lr, atol=5e-9, rtol=0.0)
        assert {"optimizer_phase", "scheduler_phase_epoch"}.issubset(log.columns)
        assert supervised.optimizer_phase.eq("supervised").all()
        assert np.array_equal(supervised.scheduler_phase_epoch.to_numpy(int), supervised.epoch.to_numpy(int))
        contrastive = log[log.stage == "contrastive"]
        if row.model in {"physiocat", "matched_no_delay"}:
            assert len(contrastive) == 5
            assert contrastive.optimizer_phase.eq("contrastive").all()
            assert np.array_equal(contrastive.scheduler_phase_epoch.to_numpy(int), np.arange(1, 6))
            assert np.allclose(
                contrastive.learning_rate.to_numpy(float),
                float(row.base_learning_rate) * np.arange(1, 6) / 5.0,
                atol=5e-9,
                rtol=0.0,
            )
        preselected = supervised[supervised.epoch < int(row.selected_epoch)]
        assert int(preselected.epochs_without_improvement.max()) < int(row.early_stopping_patience)
        assert int((preselected.epochs_without_improvement > 0).sum()) >= 1
        assert (
            supervised.loc[supervised.epoch > int(row.selected_epoch), "validation_mean_mae"]
            > float(row.selected_validation_mean_mae)
        ).all()
        if row.checkpoint_role == "representative_outer_fold":
            validation = pd.read_csv(ROOT / row.validation_predictions)
            validation_sbp = np.mean(np.abs(validation.predicted_sbp - validation.reference_sbp))
            validation_dbp = np.mean(np.abs(validation.predicted_dbp - validation.reference_dbp))
            assert len(validation) == int(row.validation_prediction_rows)
            assert validation.subject_id.nunique() == int(row.validation_prediction_subjects) == 271
            assert np.allclose(
                [validation_sbp, validation_dbp, 0.5 * (validation_sbp + validation_dbp)],
                [row.selected_validation_sbp_mae, row.selected_validation_dbp_mae, row.selected_validation_mean_mae],
                atol=5e-8,
                rtol=0.0,
            )
        assert bool(supervised.iloc[-1].early_stopping_triggered)
        assert supervised.iloc[-1].stop_reason == "early_stopping_patience_exhausted"
        assert not supervised.iloc[:-1].early_stopping_triggered.astype(bool).any()


def test_checkpoint_selection_code_is_validation_scoped():
    source = (ROOT / "scripts/train/train_fold.py").read_text(encoding="utf-8")
    assert "validation" in source.lower()
    assert "test_loader" not in source
    checkpoints = pd.read_csv(ROOT / "artifacts/checkpoints/checkpoint_manifest.csv", keep_default_na=False)
    for row in checkpoints.itertuples(index=False):
        history = pd.read_csv(ROOT / row.training_log)
        selected = history[history.selected_checkpoint.astype(bool)]
        assert len(selected) == 1
        assert np.isclose(float(selected.validation_mean_mae.iloc[0]), float(row.selected_validation_mean_mae))


def test_inference_checkpoint_serialization_is_byte_deterministic():
    (ROOT / "reports").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="checkpoint_test_", dir=ROOT / "reports") as directory:
        first = Path(directory) / "first.npz"
        second = Path(directory) / "second.npz"
        model = PhysioCAT(tiny_config())
        metadata = {"model_slug": "physiocat", "configuration_id": "unit-test", "selected_epoch": 3}
        save_inference_checkpoint(first, metadata, model.state_dict())
        save_inference_checkpoint(second, metadata, model.state_dict())
        assert first.read_bytes() == second.read_bytes()
        loaded_metadata, loaded_state = load_inference_checkpoint(first)
        assert loaded_metadata == metadata
        assert all(torch.equal(model.state_dict()[name].cpu(), loaded_state[name]) for name in loaded_state)


def test_hash_inventory_excludes_git_repository_metadata():
    script = ROOT / "scripts/reproduce/build_hash_inventory.py"
    spec = importlib.util.spec_from_file_location("build_hash_inventory_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    (ROOT / "reports").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hash_inventory_test_", dir=ROOT / "reports") as directory:
        temporary_root = Path(directory)
        (temporary_root / ".git/objects").mkdir(parents=True)
        (temporary_root / ".git/config").write_text("repository metadata", encoding="utf-8")
        (temporary_root / ".git/objects/object").write_bytes(b"git object")
        (temporary_root / "reports").mkdir()
        (temporary_root / "reports/diagnostics.txt").write_text("runtime output", encoding="utf-8")
        (temporary_root / "kept.txt").write_text("scientific artifact", encoding="utf-8")
        output, count = module.build_inventory(temporary_root)
        inventory = output.read_text(encoding="utf-8")
        assert count == 1
        assert inventory.endswith("  kept.txt\n")
        assert ".git/" not in inventory
        assert "reports/" not in inventory


def test_retention_sensitivity_and_sqi_reference_validation_are_released():
    sensitivity = pd.read_csv(ROOT / "artifacts/metrics/secondary/retention_sensitivity.csv").set_index("analysis_view")
    assert len(sensitivity) == 6
    primary_registry = pd.read_csv(ROOT / "artifacts/cohorts/cohort_registry.csv").set_index("cohort").loc["pulsedb_vital"]
    default_windows = int(sensitivity.loc["Default joint SQI (min 0.40; max 0.55)", "windows"])
    assert default_windows == int(primary_registry.windows)
    assert int(sensitivity.loc["Default joint SQI (min 0.40; max 0.55)", "subjects"]) == int(primary_registry.subjects)
    assert int(sensitivity.loc["Basic paired/target validity; no SQI threshold", "windows"]) > default_windows
    assert float(sensitivity.loc["Basic paired/target validity; no SQI threshold", "sbp_mae"]) > float(
        sensitivity.loc["Default joint SQI (min 0.40; max 0.55)", "sbp_mae"]
    )
    validation = pd.read_csv(ROOT / "artifacts/metrics/secondary/sqi_reference_validation.csv").iloc[0]
    annotations = pd.read_csv(ROOT / "artifacts/quality/sqi_validation_annotations.csv.gz")
    assert len(annotations) == int(validation.windows) == 1200
    assert float(validation.dual_rater_cohen_kappa) > 0.60
    assert float(validation.min_modality_sqi_auc) > 0.85


def test_secondary_uncertainty_names_do_not_claim_invalid_conformal_or_population_ci():
    assert not (ROOT / "artifacts/metrics/secondary/split_conformal_coverage.csv").exists()
    few_shot = pd.read_csv(ROOT / "artifacts/metrics/secondary/few_shot_calibration.csv")
    assert not any("_ci_" in column for column in few_shot.columns)
    assert any("_resampling_low" in column for column in few_shot.columns)


def test_label_source_agreement_and_complete_paired_family():
    label = pd.read_csv(ROOT / "artifacts/cohorts/label_source_audit.csv")
    primary = label[label.cohort.eq("pulsedb_vital")]
    assert len(primary) == 4
    assert primary.alternative_vs_target_mae_mmhg.max() < 0.70
    paired = pd.read_csv(ROOT / "artifacts/metrics/statistics/paired_subject_tests.csv")
    assert len(paired) == 22
    mirror = paired[paired.comparator_model.eq("ppg_leading_mirror")]
    assert set(mirror.outcome) == {"SBP", "DBP"} and len(mirror) == 2


def test_target_formation_selection_and_repository_scalar_sensitivity_are_closed():
    selection = pd.read_csv(ROOT / "artifacts/cohorts/target_formation_selection_audit.csv")
    assert set(selection.cohort) == {"pulsedb_vital", "pulsedb_mimic"}
    assert set(selection.group) == {"Target formed", "Target-formation failure"}
    assert len(selection) == 4
    failures = selection[selection.group.eq("Target-formation failure")]
    smd_columns = [column for column in selection.columns if column.endswith("_smd")]
    assert failures.repository_scalar_available_pct.min() > 65.0
    assert failures[smd_columns].abs().to_numpy(float).max() < 0.35
    assert failures[smd_columns].abs().to_numpy(float).max() > 0.01

    sensitivity = pd.read_csv(ROOT / "artifacts/metrics/secondary/repository_scalar_sensitivity.csv")
    assert set(sensitivity.cohort) == {"pulsedb_vital", "pulsedb_mimic"}
    assert set(sensitivity.model) == {"physiocat", "matched_no_delay", "mufubp_net"}
    assert set(sensitivity.outcome) == {"SBP", "DBP"}
    assert len(sensitivity) == 12
    assert sensitivity.mae_change_repository_minus_aligned.abs().max() < 0.20
    assert sensitivity.mae_change_repository_minus_aligned.abs().max() > 0.001


def test_reference_abp_quality_is_non_degenerate_and_sensitivity_is_closed():
    audit = pd.read_csv(ROOT / "data/retention/pulsedb_vital_window_audit.csv.gz")
    retained = audit.loc[audit.retained.astype(bool)]
    assert {"abp_sqi", "abp_quality_pass"}.issubset(audit.columns)
    assert int((retained.abp_sqi < 0.42).sum()) >= 500
    assert int((retained.abp_sqi >= 0.42).sum()) >= 500
    assert np.array_equal(
        retained.abp_quality_pass.astype(bool).to_numpy(),
        retained.abp_sqi.to_numpy(float) >= 0.42,
    )

    validation = pd.read_csv(
        ROOT / "artifacts/metrics/secondary/abp_reference_quality_validation.csv"
    ).iloc[0]
    annotations = pd.read_csv(
        ROOT / "artifacts/quality/abp_reference_quality_annotations.csv.gz"
    )
    assert len(annotations) == int(validation.review_windows) == 800
    assert float(validation.dual_rater_cohen_kappa) > 0.50
    assert float(validation.abp_sqi_auc) > 0.78
    assert 0.70 < float(validation.threshold_sensitivity) <= 1.0
    assert 0.60 < float(validation.threshold_specificity) <= 1.0

    sensitivity = pd.read_csv(
        ROOT / "artifacts/metrics/secondary/abp_reference_quality_sensitivity.csv"
    )
    assert len(sensitivity) == 9
    assert not sensitivity.duplicated(["analysis_view", "model"]).any()
    assert {"sbp_mae", "dbp_mae"}.issubset(sensitivity.columns)
    assert set(sensitivity.analysis_view) == {
        "All retained windows", "ABP SQI >= 0.42", "ABP SQI >= 0.55"
    }
    for _, group in sensitivity.groupby("analysis_view"):
        ordered = group.set_index("model")
        assert ordered.loc["physiocat", "sbp_mae"] < ordered.loc["mufubp_net", "sbp_mae"] < ordered.loc["matched_no_delay", "sbp_mae"]
        assert ordered.loc["physiocat", "dbp_mae"] < ordered.loc["mufubp_net", "dbp_mae"] < ordered.loc["matched_no_delay", "dbp_mae"]


def test_figure2_indices_and_random_mask_scopes_are_unambiguous():
    source = ROOT / "paper/figure_sources/diagrams/Figure_2_source.pptx"
    with zipfile.ZipFile(source) as archive:
        text = "\n".join(
            archive.read(name).decode("utf-8", errors="ignore")
            for name in archive.namelist()
            if name.endswith(".xml")
        )
    assert "j (PPG)" in text
    assert "i (ECG time)" in text
    for phrase, expected_size in {
        "ECG queries PPG": 'sz="1260"',
        "offsets 3–6 (192–384 ms centers)": 'sz="1240"',
        "PPG queries ECG": 'sz="1260"',
        "same ECG-leading pair relation": 'sz="1200"',
    }.items():
        position = text.index(phrase)
        assert expected_size in text[max(0, position - 650):position]
    supplement = (ROOT / "paper/PhysioCAT_Supplementary_Material.tex").read_text(encoding="utf-8")
    assert "fixed topology seed 82{,}000" in supplement
    assert "the stability analysis separately summarizes 20 independently trained topologies" in supplement
    manuscript = (ROOT / "paper/PhysioCAT_Manuscript.tex").read_text(encoding="utf-8")
    assert "physiological-signal fusion [39,46--48]" in manuscript
def test_signal_quality_indices_are_invariant_to_affine_amplitude_remapping():
    x = np.linspace(0.0, 8.0 * np.pi, 2000, endpoint=False)
    ecg = 0.12 * np.sin(x) + np.exp(-((np.arange(2000) % 250 - 36) / 5.0) ** 2)
    peaks = np.arange(36, 2000, 250)
    ppg = 0.8 * np.sin(x - 0.7) + 0.18 * np.sin(2.0 * x - 0.3)
    assert np.isclose(ecg_sqi(ecg, peaks), ecg_sqi(3.7 * ecg + 12.4, peaks), atol=1e-12)
    assert np.isclose(ppg_sqi(ppg), ppg_sqi(2.3 * ppg - 7.1), atol=1e-12)


def test_figure4_has_a_complete_reproducible_source_contract():
    manifest = pd.read_csv(ROOT / "artifacts/attention/figure4_panel_manifest.csv", keep_default_na=False)
    assert set(manifest.panel) == {"A", "B-C", "D", "E", "F"}
    base = ROOT / "paper/figure_sources/diagrams/Figure_4_base.pdf"
    assert base.exists() and base.stat().st_size > 100_000
    assert "schematic" in manifest.loc[manifest.panel.eq("A"), "data_role"].iloc[0]
    assert "schematic" in manifest.loc[manifest.panel.eq("F"), "data_role"].iloc[0]

    matrices = pd.read_csv(ROOT / "artifacts/attention/figure4_example_attention_matrices.csv.gz")
    assert len(matrices) == 125 * 125
    sparse = matrices.pivot(index="ecg_token", columns="ppg_token", values="physiocat_attention").to_numpy(float)
    no_delay = matrices.pivot(index="ecg_token", columns="ppg_token", values="matched_no_delay_attention").to_numpy(float)
    assert np.allclose(sparse[:-3].sum(axis=1), 1.0, atol=2e-8)
    assert np.allclose(no_delay[:-3].sum(axis=1), 1.0, atol=2e-8)
    assert np.allclose(sparse[-3:], 0.0, atol=1e-12)
    assert np.allclose(no_delay[-3:], 0.0, atol=1e-12)
    assert np.all(no_delay[:-3] > 0.0)
    rows, columns = np.nonzero(sparse > 0.0)
    assert set((columns - rows).tolist()) <= {3, 4, 5, 6}

    sparse_entropy = -(sparse[:-3] * np.log(np.clip(sparse[:-3], 1e-12, None))).sum(axis=1).mean()
    no_delay_entropy = -(no_delay[:-3] * np.log(np.clip(no_delay[:-3], 1e-12, None))).sum(axis=1).mean()
    assert no_delay_entropy > sparse_entropy + 1.0

    example = manifest[manifest.panel.eq("B-C")].iloc[0]
    assert 0.04 < float(example.no_delay_correct_band_mass) < 0.10
    assert 0.07 < float(example.no_delay_adjacent_beat_alias_mass) < 0.14
    assert 0.50 < float(example.no_delay_ecg_leading_mass) < 0.60
    assert 90.0 < float(example.no_delay_mean_effective_keys) < 118.0
    assert float(example.no_delay_beat_translation_correlation) < 0.45
    assert "fixed representative subject-grouped outer-test window" in example.selection_rule

    manuscript = (ROOT / "paper/PhysioCAT_Manuscript.tex").read_text(encoding="utf-8")
    assert "Panel A schematizes synchronized ECG, PPG, and ABP timing" in manuscript
    assert "Panel F illustrates local SQI-aware fusion" in manuscript


def test_figure_reproduction_distinguishes_science_from_renderer_bytes():
    script = (ROOT / "scripts/reproduce/reproduce_figures.py").read_text(encoding="utf-8")
    figure_7_renderer = (ROOT / "scripts/reproduce/render_figure_7.py").read_text(encoding="utf-8")
    assert '"scientific-content contract"' in script
    assert "normalized_pdf_text(submitted) == normalized_pdf_text(regenerated)" in script
    assert "geometry_delta > 1.0" in script
    assert "raster_shape_delta > 2" in script
    assert "mean_absolute_difference >= 0.04" in script
    assert "changed_fraction >= 0.18" in script
    assert '"matplotlib": matplotlib.__version__' in script
    assert '"pymupdf": fitz.VersionBind' in script
    assert '"font.family": "DejaVu Sans"' in figure_7_renderer
    assert "Arial" not in figure_7_renderer


def test_figure_reproduction_tolerates_subpoint_renderer_geometry_drift():
    import fitz

    script = ROOT / "scripts/reproduce/reproduce_figures.py"
    spec = importlib.util.spec_from_file_location("reproduce_figures_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    (ROOT / "reports").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="figure_geometry_test_", dir=ROOT / "reports") as directory:
        directory = Path(directory)
        submitted = directory / "submitted.pdf"
        regenerated = directory / "regenerated.pdf"
        for path, height in ((submitted, 200.0), (regenerated, 200.5)):
            document = fitz.open()
            page = document.new_page(width=200.0, height=height)
            page.insert_text((20.0, 20.0), "identical scientific label")
            document.save(path)
            document.close()
        result = module.verify_pdf_content(submitted, regenerated, 99)
        assert result["verification_mode"] == "scientific-content contract"
        assert result["page_geometry_max_delta_points"] <= 1.0
        assert result["raster_shape_max_delta_pixels"] <= 2


def test_figure_reproduction_rejects_material_page_geometry_change():
    import fitz

    script = ROOT / "scripts/reproduce/reproduce_figures.py"
    spec = importlib.util.spec_from_file_location("reproduce_figures_rejection_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    (ROOT / "reports").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="figure_geometry_rejection_test_", dir=ROOT / "reports") as directory:
        directory = Path(directory)
        submitted = directory / "submitted.pdf"
        regenerated = directory / "regenerated.pdf"
        for path, height in ((submitted, 200.0), (regenerated, 202.0)):
            document = fitz.open()
            page = document.new_page(width=200.0, height=height)
            page.insert_text((20.0, 20.0), "identical scientific label")
            document.save(path)
            document.close()
        with pytest.raises(AssertionError, match="page geometry changed"):
            module.verify_pdf_content(submitted, regenerated, 100)


def test_figure_reproduction_rejects_material_visual_content_change():
    import fitz

    script = ROOT / "scripts/reproduce/reproduce_figures.py"
    spec = importlib.util.spec_from_file_location("reproduce_figures_visual_rejection_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    (ROOT / "reports").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="figure_visual_rejection_test_", dir=ROOT / "reports") as directory:
        directory = Path(directory)
        submitted = directory / "submitted.pdf"
        regenerated = directory / "regenerated.pdf"
        for path, color in ((submitted, (0.1, 0.2, 0.8)), (regenerated, (0.8, 0.2, 0.1))):
            document = fitz.open()
            page = document.new_page(width=200.0, height=200.0)
            page.insert_text((20.0, 20.0), "identical scientific label")
            page.draw_rect(fitz.Rect(20.0, 40.0, 180.0, 180.0), color=color, fill=color)
            document.save(path)
            document.close()
        with pytest.raises(AssertionError, match="changed beyond the renderer-tolerance contract"):
            module.verify_pdf_content(submitted, regenerated, 101)
