"""Fine-tune Mask R-CNN on the train split. The best checkpoint is chosen on the val split only.

Usage:
    python -m models.train
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from models.data import MedBoxDataset, collate
from models.metrics import evaluate
from models.model import build_model

OUT = Path(__file__).resolve().parent / "runs"
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--limit", type=int, help="use only the first N images of each split")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    train_set = MedBoxDataset("train", train=True, limit=args.limit)
    val_set = MedBoxDataset("val", limit=args.limit)
    generator = torch.Generator().manual_seed(SEED)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=generator)

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    config = {**vars(args), "out": str(args.out), "seed": SEED, "device": str(device),
              "train_images": len(train_set), "val_images": len(val_set), "torch": torch.__version__}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets in loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val = evaluate(model, val_set, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **val})
        print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  val IoU {val['mask_iou_mean']:.4f}  "
              f"mAP@0.5 {val['mAP@0.5']:.4f}", flush=True)
        if val["mask_iou_mean"] > best:
            best = val["mask_iou_mean"]
            torch.save(model.state_dict(), args.out / "best.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val mask IoU {best:.4f}, saved to {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
