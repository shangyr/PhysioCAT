from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TensorRT FP16 or INT8 engine from the released ONNX model")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "int8"), default="fp16")
    parser.add_argument("--calibration-cache", type=Path, help="Required for INT8; generated on the target device from the fixed calibration set")
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.precision == "int8" and not args.calibration_cache:
        raise SystemExit("INT8 requires --calibration-cache")
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser_ = trt.OnnxParser(network, logger)
    if not parser_.parse(args.onnx.read_bytes()):
        messages = "\n".join(str(parser_.get_error(index)) for index in range(parser_.num_errors))
        raise RuntimeError(messages)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 1024**3))
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        class CacheCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, cache: Path):
                super().__init__()
                self.cache = cache

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                return None

            def read_calibration_cache(self):
                return self.cache.read_bytes()

            def write_calibration_cache(self, cache):
                self.cache.write_bytes(cache)

        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = CacheCalibrator(args.calibration_cache)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(serialized))
    print(args.output)


if __name__ == "__main__":
    main()
