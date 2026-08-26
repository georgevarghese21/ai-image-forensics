#!/usr/bin/env bash
# Leave-one-generator-out sweep: train one model per held-out generator.
set -e

GENS="ADM GLIDE Midjourney SD15 VQDM Wukong"

for G in $GENS; do
  echo "=============== holding out $G ==============="
  sed -i "s|^holdout_generators:.*|holdout_generators: [\"$G\"]|" configs/v1.yaml
  python -m src.train --config configs/v1.yaml --model resnet18 --tag "rn18_no_${G}"
  python -m src.evaluate --checkpoint "experiments/rn18_no_${G}/best.pt"
  cp -r experiments /content/drive/MyDrive/forensics/ 2>/dev/null || true
done

sed -i "s|^holdout_generators:.*|holdout_generators: []|" configs/v1.yaml
echo "sweep complete"
