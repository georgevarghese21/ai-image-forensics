"""FastAPI service for the forensics detector.

    uvicorn src.api.main:app --reload

GET  /health   - liveness + whether the model loaded
POST /analyse  - upload an image, get P(AI-generated)

Uploads are processed in memory only - never written to disk.
"""
import base64
import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

from src.data.dataset import to_tensor
from src.explain import grad_cam, overlay
from src.metadata import extract_from_image
from src.models.nets import build_model
from src.utils import get_device

DISCLAIMER = (
    "This is a probabilistic estimate from a single automated classifier, not "
    "proof. It can be wrong in either direction and must not be used as the "
    "sole basis for any accusation or decision."
)

CHECKPOINT_PATH = Path("models/rn18_augmented.pt")

app = FastAPI(title="AI Image Forensics API")
state = {"model": None, "crop": None, "device": None, "checkpoint_error": None}


@app.on_event("startup")
def load_model():
    device = get_device()
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model = build_model(ckpt["model"], pretrained=False).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        state["model"] = model
        state["crop"] = ckpt["crop"]
        state["device"] = device
    except FileNotFoundError:
        state["checkpoint_error"] = f"checkpoint not found at {CHECKPOINT_PATH}"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "checkpoint_error": state["checkpoint_error"],
    }


def center_crop(im, crop):
    """Same rule as ForensicDataset._crop, eval branch: center crop, pad by
    reflection if the upload is smaller than the model's input size."""
    w, h = im.size
    if w < crop or h < crop:
        im = ImageOps.expand(im, border=(max(0, (crop - w + 1) // 2),
                                          max(0, (crop - h + 1) // 2)))
        w, h = im.size
    left, top = (w - crop) // 2, (h - crop) // 2
    return im.crop((left, top, left + crop, top + crop))


@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(503, f"model not loaded: {state['checkpoint_error']}")

    raw = await file.read()
    try:
        original = Image.open(io.BytesIO(raw))
        original.load()
    except Exception:
        raise HTTPException(400, "could not read uploaded file as an image")

    # Metadata comes from the untouched upload - convert()/crop() below would
    # not necessarily preserve EXIF, and this is reported alongside the
    # verdict, never folded into it (spec: no hand-weighted fusion).
    metadata = extract_from_image(original, file.filename, len(raw))

    rgb = original.convert("RGB")
    w, h = rgb.size
    crop = state["crop"]
    padded = w < crop or h < crop
    im = center_crop(rgb, crop)

    prob, cam = grad_cam(state["model"], to_tensor(im), state["device"])

    buf = io.BytesIO()
    overlay(im, cam).save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    warnings = []
    if padded:
        # Section 6.6 of the project's findings: padding a too-small image
        # introduces reflection artefacts that resemble synthetic
        # high-frequency structure and can bias the score toward "AI".
        warnings.append(
            f"Uploaded image ({w}x{h}) is smaller than the model's "
            f"{crop}x{crop} input and was padded to fit. Padding artefacts "
            "are known to bias predictions toward 'AI-generated' - treat "
            "this result with extra caution."
        )

    return {
        "probability_ai_generated": round(prob, 4),
        "verdict": "likely AI-generated" if prob >= 0.5 else "likely real photograph",
        "metadata": metadata,
        "gradcam_png_base64": gradcam_b64,
        "reliability_warnings": warnings,
        "disclaimer": DISCLAIMER,
    }
