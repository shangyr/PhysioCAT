from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from physiocat.models import PhysioCAT


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the released PhysioCAT model to ONNX")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    model = PhysioCAT().eval()
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("state_dict", payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(1, 1, 2000), torch.zeros(1, 1, 2000), torch.ones(1, 2, 125)),
        args.output,
        input_names=["ecg", "ppg", "sqi"],
        output_names=["sbp_dbp"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(args.output)


if __name__ == "__main__":
    main()
