# Runtime profiling

The submitted timing table is sourced from `artifacts/profiling/runtime_samples.csv.gz`; environment details and the INT8 agreement check are stored beside it. Recompute its medians with:

```powershell
python scripts/reproduce/reproduce_profiling.py
```

`profile_pytorch.py`, `export_onnx.py`, and `profile_onnx.py` provide the server and CPU rerun path. `build_tensorrt_engine.py` provides the FP16 engine-build path; INT8 requires a target-device calibrator and the fixed calibration cache because TensorRT calibrators are device/runtime specific. The released INT8 agreement summary records the resulting accuracy comparison. Signal acquisition time is excluded from all post-acquisition timings.
