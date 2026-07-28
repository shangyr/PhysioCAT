from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a batch-1 PhysioCAT ONNX model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.model), providers=[args.provider])
    inputs = {
        "ecg": np.zeros((1, 1, 2000), dtype=np.float32),
        "ppg": np.zeros((1, 1, 2000), dtype=np.float32),
        "sqi": np.ones((1, 2, 125), dtype=np.float32),
    }
    for _ in range(args.warmup):
        session.run(None, inputs)
    values = []
    for index in range(args.trials):
        start = time.perf_counter_ns()
        session.run(None, inputs)
        values.append((index, (time.perf_counter_ns() - start) / 1e6))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "forward_latency_ms"])
        writer.writerows(values)
    print({"samples": len(values), "median_ms": float(np.median([value for _, value in values]))})


if __name__ == "__main__":
    main()
