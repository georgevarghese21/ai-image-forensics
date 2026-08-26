"""Evaluate a checkpoint on the test split.

    python -m src.evaluate --checkpoint experiments/rn18_debiased/best.pt

Prints headline metrics plus a per-generator breakdown. Generators absent
from training are marked NO - that column is the generalisation result.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ForensicDataset
from src.models.nets import build_model
from src.train import evaluate_loader
from src.utils import compute_metrics, get_device, load_config, set_seed


def run(model, manifest, root, crop, device, batch, generators=None):
    ds = ForensicDataset(manifest, root, split="test", generators=generators,
                         crop=crop, train=False)
    if len(ds) == 0:
        return None, None
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    return evaluate_loader(model, ld, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    manifest = Path(cfg["data"]["manifest"])
    root = manifest.parent
    crop = ckpt["crop"]
    batch = cfg["train"]["batch_size"]

    results = {"checkpoint": args.checkpoint, "model": ckpt["model"],
               "train_generators": ckpt.get("train_generators")}

    y_true, y_prob = run(model, manifest, root, crop, device, batch)
    results["overall"] = compute_metrics(y_true, y_prob)
    print("\n=== TEST SET (all generators) ===")
    for k in ("n", "accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:>10}: {results['overall'][k]}")

    # Each slice pairs ONE generator's fakes against ALL real test images, so
    # AUC stays meaningful. A generator marked NO was never seen in training.
    df = pd.read_csv(manifest)
    gens = sorted(df.query("label == 1")["generator"].unique())
    trained_on = set(ckpt.get("train_generators") or gens)

    print("\n=== PER GENERATOR ===")
    print(f"  {'generator':<14}{'seen':<7}{'n':<7}{'acc':<9}{'roc_auc'}")
    per_gen = {}
    for g in gens:
        yt, yp = run(model, manifest, root, crop, device, batch, generators=[g])
        if yt is None:
            continue
        m = compute_metrics(yt, yp)
        seen = "yes" if g in trained_on else "NO"
        per_gen[g] = {**m, "seen_in_training": g in trained_on}
        print(f"  {g:<14}{seen:<7}{m['n']:<7}{m['accuracy']:<9.4f}{m['roc_auc']:.4f}")
    results["per_generator"] = per_gen

    out = Path(args.checkpoint).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
