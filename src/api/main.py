"""FastAPI service for the forensics detector.

    uvicorn src.api.main:app --reload

GET  /health   - liveness + whether the model loaded
POST /analyse  - upload an image, get P(AI-generated)

Uploads are processed in memory only - never written to disk.
"""
import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

from src.data.dataset import to_tensor
from src.models.nets import build_model
from src.utils import get_device

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
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not read uploaded file as an image")

    im = center_crop(im, state["crop"])
    tensor = to_tensor(im).unsqueeze(0).to(state["device"])

    with torch.no_grad():
        prob = torch.sigmoid(state["model"](tensor)).item()

    return {
        "probability_ai_generated": round(prob, 4),
        "verdict": "likely AI-generated" if prob >= 0.5 else "likely real photograph",
    }
