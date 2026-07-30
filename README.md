# FractureScan AI

Full-stack web application for detecting fractures in X-ray images using the supplied fine-tuned YOLO11 model.

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Deploy on Render

1. Push this entire folder, including `models/best.pt`, to a GitHub repository.
2. In Render, select **New +** → **Blueprint** and connect the repository.
3. Render reads `render.yaml`, builds the Docker image, and gives you a public URL.

This app is for educational and research use only, not clinical diagnosis.
