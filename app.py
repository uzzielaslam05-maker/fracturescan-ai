import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="FractureScan AI", page_icon="🩻", layout="centered")

# ---------- Custom styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0B1220;
    background-image:
        linear-gradient(rgba(79, 209, 249, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(79, 209, 249, 0.04) 1px, transparent 1px);
    background-size: 32px 32px;
    color: #E6EDF7;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
footer { visibility: hidden; }

.fs-header {
    display: flex;
    align-items: baseline;
    gap: 14px;
    border-bottom: 1px solid rgba(79, 209, 249, 0.25);
    padding-bottom: 18px;
    margin-bottom: 6px;
}
.fs-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    color: #F5F9FF;
    letter-spacing: -0.02em;
}
.fs-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #4FD1F9;
    border: 1px solid rgba(79, 209, 249, 0.4);
    border-radius: 4px;
    padding: 2px 8px;
    letter-spacing: 0.05em;
}
.fs-sub {
    color: #8593AD;
    font-size: 0.95rem;
    margin-top: 8px;
    margin-bottom: 4px;
}
.fs-warn {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #F2A65A;
    background: rgba(242, 166, 90, 0.08);
    border-left: 2px solid #F2A65A;
    padding: 10px 14px;
    margin: 18px 0 26px 0;
    line-height: 1.5;
}
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(79, 209, 249, 0.35);
    border-radius: 8px;
    background: rgba(79, 209, 249, 0.03);
    padding: 6px;
}
.fs-result-card {
    background: #101A2E;
    border: 1px solid rgba(79, 209, 249, 0.2);
    border-radius: 10px;
    padding: 18px 20px;
    margin-top: 18px;
}
.fs-result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    color: #F5F9FF;
    margin-bottom: 10px;
}
.fs-detection-row {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    padding: 6px 0;
    border-bottom: 1px solid rgba(230, 237, 247, 0.06);
}
.fs-detection-row:last-child { border-bottom: none; }
.fs-label { color: #E6EDF7; }
.fs-conf { color: #4FD1F9; }
.fs-empty {
    font-family: 'JetBrains Mono', monospace;
    color: #8593AD;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="fs-header">
    <span class="fs-title">FractureScan AI</span>
    <span class="fs-tag">YOLOv8 · DETECT</span>
</div>
<div class="fs-sub">Upload an X-ray image for automated region detection.</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="fs-warn">
⚠ DEMONSTRATION MODEL — NOT A CERTIFIED DIAGNOSTIC TOOL.<br>
Do not use for real medical decisions. Consult a qualified radiologist or physician.
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input", use_container_width=True)

    with st.spinner("Running inference..."):
        results = model.predict(image, conf=0.10)
        annotated = results[0].plot()

    annotated_rgb = annotated[:, :, ::-1]
    st.image(annotated_rgb, caption="Detections", use_container_width=True)

    boxes = results[0].boxes
    st.markdown('<div class="fs-result-card">', unsafe_allow_html=True)
    st.markdown('<div class="fs-result-title">Detection Output</div>', unsafe_allow_html=True)
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.markdown(
                f'<div class="fs-detection-row"><span class="fs-label">{label}</span>'
                f'<span class="fs-conf">{conf:.2%}</span></div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown('<div class="fs-empty">No objects detected.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
