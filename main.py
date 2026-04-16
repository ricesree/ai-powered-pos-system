from __future__ import annotations

import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model_utils import image_to_tensor, load_vegetable_checkpoint

app = FastAPI()
MODEL_PATH = Path("model") / "vegetable_cnn.pt"

model_state = {
    "loaded": False,
    "model": None,
    "class_names": [],
    "image_size": 128,
    "device": "cpu",
    "message": "",
}


def load_model() -> None:
    if not MODEL_PATH.exists():
        model_state["loaded"] = False
        model_state["message"] = f"Model not found at {MODEL_PATH}. Train once, then reuse this file directly."
        return

    loaded = load_vegetable_checkpoint(MODEL_PATH)
    class_names = loaded["class_names"]

    model_state["loaded"] = True
    model_state["model"] = loaded["model"]
    model_state["class_names"] = class_names
    model_state["image_size"] = loaded["image_size"]
    model_state["device"] = loaded["device"]
    model_state["message"] = (
        f"Saved model loaded successfully with {len(class_names)} classes from {MODEL_PATH}."
    )


def predict_image(image: Image.Image) -> tuple[str, float, list[dict[str, float | str]]]:
    model = model_state["model"]
    class_names = model_state["class_names"]
    image_size = model_state["image_size"]
    device = torch.device(model_state["device"])

    tensor = image_to_tensor(image, image_size=image_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_k = min(3, probs.shape[1])
        confidences, indices = torch.topk(probs, k=top_k, dim=1)

    top_predictions: list[dict[str, float | str]] = []
    for score, idx in zip(confidences[0], indices[0]):
        top_predictions.append(
            {
                "label": class_names[int(idx.item())],
                "confidence": round(float(score.item()), 4),
            }
        )

    best = top_predictions[0]
    return str(best["label"]), float(best["confidence"]), top_predictions


# Allow Electron to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_model()


@app.get("/")
async def root():
    return {
        "status": "ok",
        "model_loaded": model_state["loaded"],
        "device": model_state["device"],
        "message": model_state["message"],
        "usage": "Use POST /predict with multipart form-data key 'file'.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_state["loaded"]:
        return {"status": "error", "message": model_state["message"]}

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    prediction, confidence, top_predictions = predict_image(image)
    return {
        "status": "ok",
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "predictions": [item["label"] for item in top_predictions],
        "top_predictions": top_predictions,
    }


@app.post("/reload-model")
async def reload_model():
    load_model()
    return {
        "status": "ok",
        "model_loaded": model_state["loaded"],
        "device": model_state["device"],
        "message": model_state["message"],
    }
