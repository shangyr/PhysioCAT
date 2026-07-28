from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from physiocat.models import PhysioCAT


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile batch-1 PhysioCAT PyTorch inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
        raise SystemExit("FP16 profiling requires a CUDA device")
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    torch.manual_seed(42)
    model = PhysioCAT().eval().to(device=device, dtype=dtype)
    ecg = torch.randn(1, 1, 2000, device=device, dtype=dtype)
    ppg = torch.randn(1, 1, 2000, device=device, dtype=dtype)
    sqi = torch.rand(1, 2, 125, device=device, dtype=dtype)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(ecg, ppg, sqi)
        synchronize()
        values = []
        for index in range(args.trials):
            start = time.perf_counter_ns()
            model(ecg, ppg, sqi)
            synchronize()
            values.append((index, (time.perf_counter_ns() - start) / 1e6))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "forward_latency_ms"])
        writer.writerows(values)
    print({"samples": len(values), "median_ms": float(np.median([value for _, value in values]))})


if __name__ == "__main__":
    main()
