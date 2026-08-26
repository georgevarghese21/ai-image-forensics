"""Grad-CAM: visualise which image regions influenced the prediction.

    python -m src.explain --checkpoint experiments/rn18_augmented/best.pt \
        --image path/to/image.jpg --out heatmap.png

This shows where the model was SENSITIVE, not why it decided. A 7x7 activation
map upscaled to 224x224 indicates regions, not pixels, and does not constitute
an explanation of the model's reasoning.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import to_tensor
from src.models.nets import build_model
from src.utils import get_device


def last_conv_layer(model):
    """Find the final convolutional layer, whatever the architecture."""
    target = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target = module
    if target is None:
        raise SystemExit("no Conv2d layer found")
    return target


def grad_cam(model, tensor, device):
    """Return (probability, 224x224 heatmap normalised to 0-1)."""
    activations, gradients = {}, {}
    layer = last_conv_layer(model)

    def fwd_hook(_m, _i, out):
        activations["value"] = out.detach()

    def bwd_hook(_m, _gi, gout):
        gradients["value"] = gout[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    x = tensor.unsqueeze(0).to(device)
    logit = model(x)
    prob = torch.sigmoid(logit).item()
    logit.backward()          # gradient of the score w.r.t. the feature maps

    h1.remove()
    h2.remove()

    acts = activations["value"][0]      # (C, H, W)
    grads = gradients["value"][0]       # (C, H, W)

    # Each channel's importance is its mean gradient: how much raising that
    # feature map would raise the score.
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=0))   # keep positive contributions

    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(cam, size=(tensor.shape[1], tensor.shape[2]),
                        mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return prob, cam


def overlay(im, cam, alpha=0.5):
    """Blend a red-blue heatmap over the image."""
    base = np.asarray(im, dtype=np.float32) / 255.0
    heat = np.zeros_like(base)
    heat[..., 0] = cam            # red where the model looked
    heat[..., 2] = 1.0 - cam      # blue where it did not
    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="gradcam.png")
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    crop = ckpt["crop"]
    with Image.open(args.image) as raw:
        im = raw.convert("RGB")
        w, h = im.size
        im = im.crop(((w - crop) // 2, (h - crop) // 2,
                      (w - crop) // 2 + crop, (h - crop) // 2 + crop))

    prob, cam = grad_cam(model, to_tensor(im), device)

    combined = Image.new("RGB", (crop * 2, crop))
    combined.paste(im, (0, 0))
    combined.paste(overlay(im, cam), (crop, 0))
    combined.save(args.out)

    verdict = "likely AI-generated" if prob >= 0.5 else "likely real photograph"
    print(f"P(AI-generated) = {prob:.4f}  ->  {verdict}")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
