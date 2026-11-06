"""Evaluate a checkpoint on the test split. Run once, after the checkpoint is chosen on val.

Usage:
    python -m models.evaluate
"""
import argparse
import json
from pathlib import Path

import torch

from models.data import MedBoxDataset
from models.metrics import evaluate
from models.model import load_checkpoint

RUNS = Path(__file__).resolve().parent / "runs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=RUNS / "best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=RUNS / "test_metrics.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate(load_checkpoint(args.weights, device), MedBoxDataset(args.split), device)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
