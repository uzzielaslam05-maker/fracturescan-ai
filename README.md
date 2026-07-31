# FractureScan AI — X-Ray Object Detection

A Streamlit web app that uses a YOLOv8 model to detect and label objects (e.g. fractures/anomalies) in X-ray images.

⚠️ **Disclaimer:** This is a demonstration project, not a certified medical diagnostic tool. Do not use it to make real medical decisions — always consult a qualified doctor or radiologist.

## How it works

1. Upload an X-ray image (JPG or PNG).
2. The model (`best.pt`, a YOLOv8 detection model) runs inference on the image.
3. Detected regions are drawn as bounding boxes on the image, along with the predicted label and confidence score.

## Tech stack

- **Streamlit** — web app framework
- **Ultralytics YOLOv8** — object detection model
- **OpenCV (headless)** — image processing
- **Pillow** — image loading

## Project structure

```
├── app.py              # Streamlit app
├── best.pt              # YOLOv8 model weights
├── requirements.txt      # Python dependencies
└── packages.txt          # System-level dependencies (for Streamlit Cloud)
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live demo

Deployed on Streamlit Community Cloud — https://uzzielaslam05-maker-fracturescan-ai-app-kitqwt.streamlit.app/

## Notes on accuracy

Model performance depends on the size and quality of the training dataset used. Treat results as illustrative rather than authoritative.
