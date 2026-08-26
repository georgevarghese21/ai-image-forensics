"""Summarise the leave-one-generator-out sweep.

For each generator G, compare G's test AUC in two models:
  - the baseline, which trained on all generators including G
  - the holdout model, which never saw G

Same test images and architecture both times, so the difference isolates the
effect of G being unseen. Caveat: holdout models also train on one fewer
generator, so part of any drop is reduced training data, not novelty.
"""
import json
from pathlib import Path

base = json.loads(Path("experiments/rn18_debiased/results.json").read_text())["per_generator"]

rows = []
for d in sorted(Path("experiments").glob("rn18_no_*")):
    f = d / "results.json"
    if not f.exists():
        continue
    held = d.name.replace("rn18_no_", "")
    per = json.loads(f.read_text())["per_generator"]
    if held not in per or held not in base:
        continue

    rows.append({
        "generator": held,
        "seen_auc": base[held]["roc_auc"],       # baseline: G was in training
        "unseen_auc": per[held]["roc_auc"],      # holdout: G never seen
        "others_auc": sum(v["roc_auc"] for k, v in per.items() if k != held)
                      / (len(per) - 1),          # same model, generators it did see
    })

hdr = f"{'generator':<14}{'seen AUC':<12}{'unseen AUC':<13}{'drop':<11}{'others (same model)'}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    drop = r["seen_auc"] - r["unseen_auc"]
    print(f"{r['generator']:<14}{r['seen_auc']:<12.4f}{r['unseen_auc']:<13.4f}"
          f"{drop:<+11.4f}{r['others_auc']:.4f}")

if rows:
    drops = [r["seen_auc"] - r["unseen_auc"] for r in rows]
    print(f"\nmean drop: {sum(drops) / len(drops):+.4f}   "
          f"worst: {max(drops):+.4f}   best: {min(drops):+.4f}")
    Path("experiments/logo_summary.json").write_text(json.dumps(rows, indent=2))
    print("written -> experiments/logo_summary.json")
