# Reviewer guide

Run the complete audit with:

```bash
python scripts/reproduce/reproduce_all.py
python -m pytest -q tests
```

The main inspection routes are:

| Question | Files |
|---|---|
| Main and external results | `artifacts/predictions/`, `scripts/reproduce/reproduce_main_tables.py` |
| Subject-disjoint and random-segment protocols | `data/folds/`, `scripts/reproduce/verify_fold_membership.py`, `scripts/train/run_random_segment_cv.py`, `artifacts/logs/training/model_configuration_registry.csv` |
| Training and prediction evidence | `model_configuration_registry.csv`, complete five-fold manifests, `prediction_authority_manifest.csv`, formal/stability/random-mask inference ledgers, released checkpoints, compute audit, and `verify_training_lineage.py` |
| Prediction authorities | `artifacts/predictions/prediction_authority_manifest.csv`, complete prediction tables |
| Retention, fields, and labels | `data/retention/`, `artifacts/provenance/dataset_field_and_label_contract.csv`, `src/physiocat/preprocessing.py` |
| Reference-target construction | `artifacts/cohorts/label_source_audit.csv`, `artifacts/cohorts/target_formation_selection_audit.csv`, `artifacts/metrics/secondary/repository_scalar_sensitivity.csv`, `abp_reference_quality_sensitivity.csv`, `abp_reference_quality_validation.csv`, `artifacts/quality/abp_reference_quality_annotations.csv.gz`, `data/retention/*_raw_candidate_manifest.csv.gz`, `data/retention/*_window_audit.csv.gz`, `src/physiocat/preprocessing.py`, `src/physiocat/adapters.py` |
| PhysioCAT implementation | `src/physiocat/models.py`, `configs/model/` |
| Baseline implementations | `src/physiocat/baselines.py`, `configs/model/` |
| Fixed-patch delay implementation and mechanism controls | `paper/PhysioCAT_Manuscript.tex`, `src/physiocat/models.py`, `artifacts/metrics/mechanism/`, `artifacts/attention/`; fixed patch-timestamp masks, sparse attention summaries, structured controls, and descriptive apparent-PAT analysis |
| Waveform perturbation definitions | `src/physiocat/perturbations.py`, `artifacts/protocol/timing_perturbation_protocol.csv`, `artifacts/metrics/secondary/negative_controls.csv`, `artifacts/metrics/secondary/inference_control_lineage.csv`; token-aligned circular shifts move complete 16-sample patches, preserve within-token samples, energy, and Fourier magnitude, and use the fixed 3--10-token range |
| Attention export integrity | `scripts/reproduce/verify_attention_export.py`, `scripts/reproduce/reproduce_attention.py` |
| Statistics | `artifacts/metrics/statistics/`, `scripts/reproduce/reproduce_statistics.py` |
| Pressure-range and residual structure | `artifacts/metrics/secondary/conditional_bp_performance.csv`, `calibration_diagnostics.csv`, `error_tail_composition.csv`, `artifacts/metrics/statistics/repeated_measures_agreement.csv` |
| MIMIC target-protocol identity | `artifacts/cohorts/mimic_ecosystem_identity_audit.csv`, `scripts/data/canonicalize_candidate_lineage.py`, `artifacts/metrics/secondary/mimic_bp_protocol_audit.csv`, `artifacts/protocol/external_prediction_manifest.csv`; both protocols use the shared `mimic-iii` hash namespace |
| Submitted text and figures | `paper/`, `scripts/reproduce/render_figure_5.py`, `render_figure_6.py`, `render_figure_7.py` |
| Exact validated environment | `requirements/requirements-lock.txt` |

The reviewer package uses one complete five-fold subject-grouped OOF evidence chain for all 50 reported configurations. The formal ledger contains 250 fits, including 100 fits for 20 independently trained degree-preserving random-rewiring topologies; the stability ledger contains 40 additional fits from two extra seeds for four mechanism-relevant models. Each random graph preserves both endpoint degree vectors and the reverse branch is its transpose. `design_parameter_provenance.csv` separates design-fixed choices from a common three-candidate neural training-setting grid, and `fold_local_training_setting_selection.csv` verifies that selection used only each fold's validation subjects. A separate source-model ledger contains nine source-training runs (three models by three seeds), with the primary-seed checkpoints released and every seed bound to source-validation and zero-shot target predictions. Both matched core checkpoints are released for every outer fold with complete validation-prediction shards, selected-to-stop histories, checkpoints, and executable replay tensors. `configuration_validation_test_summary.csv`, `core_validation_test_consistency.json`, and the compute summaries expose campaign-level validation/test and resource evidence directly. Phase-specific histories verify that contrastive and supervised optimization each starts with a fresh optimizer and warm-up schedule. Compact architecture fixtures remain separately scoped; the released training scripts expose the executable schedules and data routes for locally obtained source data.
