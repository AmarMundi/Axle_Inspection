import os
import io
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
import cv2

@st.cache_resource(show_spinner=False)
def load_model(weights_choice: str, custom_path: str):
    """
    weights_choice: "demo" -> use yolov8n.pt (auto-downloads from Ultralytics)
                    "custom" -> use custom_path
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        st.error(
            "Ultralytics (YOLO) is not installed. Install it with "
            "`pip install ultralytics`.\\n"
            f"Import error: {e}"
        )
        return None, "cpu"

    # Device selection: CUDA, MPS (Apple Silicon), else CPU
    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"  # Apple Silicon (Metal)
        elif torch.cuda.is_available():
            device = "cuda:0"
    except Exception:
        pass

    model_path = "yolov8n.pt" if weights_choice == "demo" else custom_path
    model = YOLO(model_path)
    return model, device

def annotate_result(res):
    annotated_bgr = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    det_rows = []
    names = res.names
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            det_rows.append({
                "class": names.get(cls[i], str(cls[i])),
                "confidence": float(conf[i]),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
            })
    return annotated_rgb, det_rows

def main():
    st.set_page_config(page_title="Object Detection – Mac & Jetson-ready", layout="wide")
    st.title("📦 Object Detection Demo")
    st.caption(
        "Upload an image to see predictions from a YOLO model. "
        "Choose the built-in **YOLOv8n** demo or your own weights."
    )

    with st.sidebar:
        st.header("Model")
        weights_choice = st.radio(
            "Weights source",
            ["Use built-in YOLOv8n (demo)", "Use my local weights"],
            index=0
        )
        use_demo = weights_choice.startswith("Use built-in")
        custom_path = st.text_input(
            "Path to model file",
            value="best.pt",
            disabled=use_demo,
            help="For your own model (.pt/.onnx/.engine)."
        )
        conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
        iou = st.slider("NMS IoU", 0.0, 1.0, 0.45, 0.01)
        imgsz = st.select_slider("Image size (inference)", options=[320, 416, 480, 512, 640], value=640)
        agnostic = st.checkbox("Class-agnostic NMS", value=False)
        save_annot = st.checkbox("Enable annotated image download", value=True)
        st.markdown("---")
        st.write("💡 Apple Silicon users: PyTorch will run on **MPS** if available.")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    model, device = load_model("demo" if use_demo else "custom", custom_path)
    if model is None:
        return

    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(img, use_container_width=True)

    # Inference
    st.subheader("Prediction")
    with st.spinner("Running inference..."):
        use_half = device.startswith("cuda")  # only CUDA supports half precision
        results = model.predict(
            source=np.array(img),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            agnostic_nms=agnostic,
            verbose=False,
            half=use_half
        )

    res = results[0]
    annotated_rgb, rows = annotate_result(res)

    with col2:
        st.subheader("Annotated")
        st.image(annotated_rgb, use_container_width=True)
        if save_annot:
            buf = io.BytesIO()
            Image.fromarray(annotated_rgb).save(buf, format="PNG")
            st.download_button(
                "⬇️ Download annotated image",
                data=buf.getvalue(),
                file_name="prediction.png",
                mime="image/png"
            )

    st.markdown("### Detections")
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No objects detected with the current thresholds.")

    st.markdown("---")
    st.caption(
        "Run: `streamlit run app.py`  \n"
        "If Ultralytics needs the demo weights, it will download `yolov8n.pt` automatically."
    )

if __name__ == "__main__":
    main()
