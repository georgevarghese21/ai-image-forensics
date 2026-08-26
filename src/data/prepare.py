"""Build a bias-controlled dataset manifest.

The raw data has two giveaways: real images are JPEG, every generator is PNG;
and each generator has a fixed size while real photos vary. A model would
learn those instead of generation artefacts. Every image therefore goes
through identical treatment - same centre crop, same JPEG quality.

Run with --no-debias to build the confounded version for the ablation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def find_images(source):
    """List every image under source/real/ and source/ai/<generator>/."""
    source = Path(source)
    records = []

    for item in (source / "real").rglob("*"):
        if item.suffix.lower() not in IMG_EXTS:
            continue
        records.append({"src": item, "label": 0, "generator": "Real"})

    for gen_dir in sorted(p for p in (source / "ai").iterdir() if p.is_dir()):
        for item in gen_dir.rglob("*"):
            if item.suffix.lower() not in IMG_EXTS:
                continue
            records.append({"src": item, "label": 1, "generator": gen_dir.name})

    return records


def group_key(path):
    """Unique key per file.

    Tiny-GenImage carries no content labels, so images cannot be grouped by
    scene. Every file is its own group; recorded as a README limitation.
    """
    return hashlib.md5(str(path).encode()).hexdigest()[:12]


def process_image(src, dst, crop=256, quality=95, debias=True):
    """Standardise one image. Returns False if it was skipped."""
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size

            if min(w, h) < crop:
                return False

            if debias:
                left = (w - crop) // 2
                top = (h - crop) // 2
                im = im.crop((left, top, left + crop, top + crop))

            dst = dst.with_suffix(".jpg")
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="JPEG", quality=quality, subsampling=0)
        return True
    except Exception:
        return False


def balance(df, seed=42):
    """Equal count per generator, with the real class matched to the AI total."""
    ai = df[df["label"] == 1]
    real = df[df["label"] == 0]

    n_gens = ai["generator"].nunique()
    smallest_gen = ai.groupby("generator").size().min()
    per_gen = min(smallest_gen, len(real) // n_gens)

    if per_gen == 0:
        raise SystemExit("too few real images to balance against the generators")

    parts = [g.sample(per_gen, random_state=seed) for _, g in ai.groupby("generator")]
    ai_bal = pd.concat(parts).reset_index(drop=True)
    real_bal = real.sample(len(ai_bal), random_state=seed)

    return pd.concat([real_bal, ai_bal]).reset_index(drop=True)


def assign_splits(df, seed=42):
    """70/15/15, split by group so no group spans two splits."""
    groups = sorted(df["group"].unique())
    shuffled = pd.Series(groups).sample(frac=1.0, random_state=seed).tolist()

    n = len(shuffled)
    n_train, n_val = int(0.70 * n), int(0.15 * n)

    lookup = {}
    for i, g in enumerate(shuffled):
        if i < n_train:
            lookup[g] = "train"
        elif i < n_train + n_val:
            lookup[g] = "val"
        else:
            lookup[g] = "test"

    df = df.copy()
    df["split"] = df["group"].map(lookup)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--no-debias", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    debias = not args.no_debias

    records = find_images(args.source)
    print(f"found {len(records)} images")

    kept = []
    for rec in records:
        src = rec["src"]
        sub = "real" if rec["label"] == 0 else f"ai/{rec['generator']}"
        dst = out / sub / src.name

        if not process_image(src, dst, args.crop, args.quality, debias):
            continue

        kept.append({
            "path": str(dst.with_suffix(".jpg").relative_to(out)),
            "label": rec["label"],
            "generator": rec["generator"],
            "group": group_key(src),
        })

    print(f"kept {len(kept)} after size filtering")
    if not kept:
        raise SystemExit("no images survived - check --source and --crop")

    df = assign_splits(balance(pd.DataFrame(kept), args.seed), args.seed)

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "manifest.csv", index=False)

    summary = {
        "debiased": debias,
        "crop": args.crop,
        "jpeg_quality": args.quality,
        "total": len(df),
        "by_split": df.groupby("split").size().to_dict(),
        "by_generator": df.groupby("generator").size().to_dict(),
        "real_fraction": round(float((df.label == 0).mean()), 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


