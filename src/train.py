"""Train one detector.

    python -m src.train --config configs/v1.yaml --model resnet18 --tag rn18
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ForensicDataset
from src.models.nets import build_model
from src.utils import compute_metrics, get_device, load_config, set_seed


@torch.no_grad()   # measuring, not learning - skip gradient tracking
def evaluate_loader(model, loader, device):
    """Run the model over a loader, return (true labels, probabilities)."""
    model.eval()   # dropout off, batchnorm uses stored stats
    probs, labels = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        logits = model(x.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--augment", action="store_true",
                    help="override config: force augmentation on")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    model_name = args.model or cfg["model"]["name"]
    tag = args.tag or model_name
    augment = args.augment or cfg["data"].get("augment", False)

    manifest = Path(cfg["data"]["manifest"])
    root = manifest.parent
    crop = cfg["data"]["crop"]

    # Leave-one-generator-out: exclude the held-out generators from training
    # and validation entirely, so the test on them is genuinely unseen.
    holdout = cfg.get("holdout_generators") or []
    train_gens = None
    if holdout:
        all_gens = set(pd.read_csv(manifest).query("label == 1")["generator"].unique())
        train_gens = sorted(all_gens - set(holdout))
        print(f"holding out {holdout} | training on {train_gens}")

    # Validation stays clean and deterministic - augmenting it would make
    # scores move for reasons unrelated to the model.
    train_ds = ForensicDataset(manifest, root, split="train", generators=train_gens,
                               crop=crop, train=True, augment=augment)
    val_ds = ForensicDataset(manifest, root, split="val", generators=train_gens,
                             crop=crop, train=False)
    if len(train_ds) == 0:
        raise SystemExit("empty training set - check the manifest path")
    print(f"train {len(train_ds)} | val {len(val_ds)} | device {device} | augment {augment}")

    nw = cfg["data"]["num_workers"]
    bs = cfg["train"]["batch_size"]
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                          pin_memory=(device.type == "cuda"), drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    model = build_model(model_name, cfg["model"]["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    criterion = nn.BCEWithLogitsLoss()

    out_dir = Path(cfg["output_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_auc, patience = [], -1.0, 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running, t0 = 0.0, time.time()

        for x, y in tqdm(train_ld, desc=f"epoch {epoch + 1}/{cfg['train']['epochs']}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)   # gradients accumulate by default
            loss = criterion(model(x), y)     # how wrong were the guesses
            loss.backward()                   # which way should each weight move
            opt.step()                        # move them
            running += loss.item() * x.size(0)

        sched.step()

        y_true, y_prob = evaluate_loader(model, val_ld, device)
        m = compute_metrics(y_true, y_prob)
        m.update(epoch=epoch + 1, train_loss=running / len(train_ds),
                 seconds=round(time.time() - t0, 1))
        history.append(m)
        print(f"  loss {m['train_loss']:.4f} | val acc {m['accuracy']:.4f} "
              f"| val AUC {m['roc_auc']:.4f}")

        # Select on ROC-AUC, not accuracy: threshold-free, and immune to the
        # class-ratio drift that group-based splitting introduces.
        if m["roc_auc"] > best_auc:
            best_auc, patience = m["roc_auc"], 0
            torch.save({"model": model_name, "state_dict": model.state_dict(),
                        "crop": crop, "val_roc_auc": best_auc,
                        "train_generators": train_gens, "augment": augment},
                       out_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print(f"early stop at epoch {epoch + 1}")
                break

    (out_dir / "history.json").write_text(json.dumps(
        {"tag": tag, "model": model_name, "config": cfg, "holdout": holdout,
         "augment": augment, "best_val_roc_auc": best_auc, "history": history},
        indent=2))
    print(f"\nbest val ROC-AUC {best_auc:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
