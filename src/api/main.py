"""FastAPI service for the forensics detector.

    uvicorn src.api.main:app --reload

GET  /health   - liveness + whether the model loaded
POST /analyse  - not yet implemented (next step)
"""
from pathlib import Path

import torch
from fastapi import FastAPI

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
