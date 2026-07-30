from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "best.pt"))
MAX_FILE_SIZE = 10 * 1024 * 1024
model: YOLO | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    yield
    model = None


app = FastAPI(title="FractureScan AI", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ready" if model else "loading", "model": "YOLO11 Fracture Detector"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), confidence: float = 0.25):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "Please upload an image file.")
    if not 0.05 <= confidence <= 0.95:
        raise HTTPException(422, "Confidence must be between 0.05 and 0.95.")
    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(413, "Image must be 10 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(422, "This image could not be processed.") from exc
    if model is None:
        raise HTTPException(503, "The model is still loading. Please retry shortly.")
    result = model.predict(image, conf=confidence, verbose=False)[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = [round(float(value), 1) for value in box.xyxy[0].tolist()]
            detections.append({"label": result.names[class_id], "confidence": round(float(box.conf[0].item()) * 100, 1), "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}})
    return JSONResponse({"detections": detections, "count": len(detections), "image": {"width": image.width, "height": image.height}})
