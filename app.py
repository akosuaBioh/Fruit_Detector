"""
Fruit Detection App
Group 5 - AIMS Kigali
Uses YOLOv8 trained on 6 fruit classes: apple, kiwi, orange, pear, strawberry, tomato
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# ── Page config ──
st.set_page_config(
    page_title="Fruit Detector",
    page_icon="🍎",
    layout="wide"
)

# ── Custom styling ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 600;
        color: #2D2B3D;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        font-size: 1.05rem;
        color: #6B6880;
        margin-top: 0;
    }

    .class-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 4px;
        color: white;
    }

    .metric-card {
        background: #F8F7FC;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #E8E6F0;
    }

    .metric-card h3 {
        color: #6B6880;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    .metric-card p {
        color: #2D2B3D;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
    }

    .detection-box {
        background: #F0FFF4;
        border: 1px solid #C6F6D5;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }

    .no-detection-box {
        background: #FFF5F5;
        border: 1px solid #FED7D7;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }

    hr {
        border: none;
        border-top: 1px solid #E8E6F0;
        margin: 1.5rem 0;
    }

    .footer {
        text-align: center;
        color: #A0A0B0;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Class colors ──
CLASS_COLORS = {
    "apple": "#E53E3E",
    "kiwi": "#38A169",
    "orange": "#DD6B20",
    "pear": "#D69E2E",
    "strawberry": "#D53F8C",
    "tomato": "#C53030"
}

# ── Header ──
st.markdown("""
<div class="main-header">
    <h1>🍎 Fruit Detector</h1>
    <p>Upload a fruit image and let our YOLOv8 model identify what is inside.</p>
</div>
""", unsafe_allow_html=True)

# ── Supported classes display ──
pills_html = ""
for cls, color in CLASS_COLORS.items():
    pills_html += f'<span class="class-pill" style="background-color: {color};">{cls}</span>'

st.markdown(f"""
<div style="text-align: center; margin-bottom: 1.5rem;">
    <p style="color: #6B6880; font-size: 0.9rem; margin-bottom: 0.5rem;">Trained to detect</p>
    {pills_html}
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Load model ──
@st.cache_resource
def load_model():
    """Load the trained YOLOv8 model."""
    return YOLO("best.pt")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model. Make sure `best.pt` is in the same folder as this script. Error: {e}")

# ── Main layout ──
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### Upload an image")

    uploaded_file = st.file_uploader(
        "Choose a fruit image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Only show detections above this confidence level."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

with col_result:
    st.markdown("### Detection results")

    if uploaded_file is not None and model_loaded:
        # run detection
        img_array = np.array(image)
        results = model.predict(source=img_array, conf=confidence, verbose=False)
        result = results[0]

        # draw boxes on image
        annotated = result.plot()
        annotated_rgb = annotated
        st.image(annotated_rgb, caption="Detected fruits", use_container_width=True)

        # count detections
        boxes = result.boxes
        n_detections = len(boxes)

        if n_detections > 0:
            # gather stats
            classes_detected = []
            confidences = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = result.names[cls_id]
                classes_detected.append(cls_name)
                confidences.append(conf)

            avg_conf = np.mean(confidences) * 100

            # summary
            st.markdown(f"""
            <div class="detection-box">
                Found <strong>{n_detections}</strong> fruit(s) with an average confidence of <strong>{avg_conf:.1f}%</strong>.
            </div>
            """, unsafe_allow_html=True)

            # metrics row
            unique_classes = list(set(classes_detected))
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Fruits found</h3>
                    <p>{n_detections}</p>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Classes</h3>
                    <p>{len(unique_classes)}</p>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg confidence</h3>
                    <p>{avg_conf:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # per-detection table
            st.markdown("")
            st.markdown("**Detection details**")
            import pandas as pd
            det_data = []
            for cls_name, conf in zip(classes_detected, confidences):
                det_data.append({"Fruit": cls_name, "Confidence": f"{conf*100:.1f}%"})
            st.dataframe(pd.DataFrame(det_data), use_container_width=True, hide_index=True)

            # download button
            result_image = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            st.download_button(
                label="Download result image",
                data=buf.getvalue(),
                file_name="detection_result.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.markdown("""
            <div class="no-detection-box">
                No fruits detected. Try a different image or lower the confidence threshold.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #A0A0B0;">
            <p style="font-size: 2rem;">🖼️</p>
            <p>Upload an image to see results here.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    Fruit Detector &middot; Group 5 &middot; AIMS Kigali &middot; Powered by YOLOv8
</div>
""", unsafe_allow_html=True)
