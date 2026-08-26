"""Evaluate a checkpoint on the test split.

    python -m src.evaluate --checkpoint experiments/rn18_debiased/best.pt
    python -m src.evaluate --checkpoint experiments/rn18_debiased/best.pt --robustness

Prints headline metrics, a per-generator breakdown (generators absent from
training are marked NO), and optionally a robustness sweep.
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

# Eval-time degradations standing in for real-world image handling.
PERTURBATIONS = {
    "original": {},
    "jpeg_90": {"jpeg": 90},
    "jpeg_70": {"jpeg": 70},
    "jpeg_50": {"jpeg": 50},
    "jpeg_30": {"jpeg": 30},
    "resize_75": {"scale": 0.75},
    "resize_50": {"scale": 0.50},
    "blur_1.5": {"blur": 1.5},
    "screenshot": {"scale": 0.8, "jpeg": 60},
}


def run(model, manifest, root, crop, device, batch, generators=None, perturbation=None):
    ds = ForensicDataset(manifest, root, split="test", generators=generators,
                         crop=crop, train=False, perturbation=perturbation)
    if len(ds) == 0:
        return None, None
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    return evaluate_loader(model, ld, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--robustness", action="store_true")
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

    if args.robustness:
        print("\n=== ROBUSTNESS ===")
        print(f"  {'transformation':<16}{'acc':<9}{'roc_auc'}")
        sweep = {}
        for name, pert in PERTURBATIONS.items():
            yt, yp = run(model, manifest, root, crop, device, batch, perturbation=pert)
            m = compute_metrics(yt, yp)
            sweep[name] = m
            print(f"  {name:<16}{m['accuracy']:<9.4f}{m['roc_auc']:.4f}")
        results["robustness"] = sweep

    out = Path(args.checkpoint).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
