import streamlit as st
import cv2
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from one_vision import process_frame, draw_boxes, log_detections
import numpy as np
import os
import json
import pandas as pd

# Streamlit page config
st.set_page_config(page_title="Zero-Shot Object Detection", layout="wide")

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
if "frame_id" not in st.session_state:
    st.session_state.frame_id = 0

def main():
    st.title("Zero-Shot Object Detection with OWL-ViT")

    # Sidebar for inputs
    st.sidebar.header("Configuration")
    video_source_type = st.sidebar.selectbox("Video Source", ["Webcam", "Upload Video File"])
    
    if video_source_type == "Webcam":
        video_source = 0  # Default webcam index
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi"])
        if uploaded_file is not None:
            # Save uploaded file temporarily
            video_path = os.path.join("temp_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            video_source = video_path
        else:
            video_source = None

    # Prompts input
    prompts_input = st.sidebar.text_input(
        "Objects to Detect (comma-separated)",
        value="lightbulb,matchstick,monitor,lion,gaming console"
    )
    text_prompts = [p.strip() for p in prompts_input.split(",") if p.strip()]

    # Start/Stop buttons
    start_button = st.sidebar.button("Start Detection")
    stop_button = st.sidebar.button("Stop Detection")

    # Load model and processor
    try:
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return

    # Video display placeholder
    video_placeholder = st.empty()

    # Log display
    st.subheader("Detection Logs")
    log_placeholder = st.empty()

    # Handle Start/Stop
    if start_button:
        st.session_state.running = True
        st.session_state.frame_id = 0

    if stop_button:
        st.session_state.running = False

    # Initialize video capture
    if st.session_state.running and video_source is not None:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error("Error: Could not open video source.")
            st.session_state.running = False
            return

        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("End of video or error reading frame.")
                break

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))

            # Process frame
            boxes, scores, labels = process_frame(frame, text_prompts, model, processor, device)

            # Draw bounding boxes
            frame = draw_boxes(frame, boxes, scores, labels, text_prompts)

            # Log detections
            log_detections(st.session_state.frame_id, boxes, scores, labels, text_prompts)

            # Convert frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display frame
            video_placeholder.image(frame_rgb, caption="Live Detection", use_column_width=True)

            # Update frame ID
            st.session_state.frame_id += 1

            # Display logs
            if os.path.exists("detections.json"):
                with open("detections.json", "r") as f:
                    logs = [json.loads(line) for line in f]
                if logs:
                    log_df = pd.json_normalize(
                        [det for log in logs for det in log["detections"]],
                        meta=["frame"]
                    )
                    log_placeholder.dataframe(log_df)

        # Cleanup
        cap.release()
        if video_source_type == "Upload Video File" and os.path.exists(video_source):
            os.remove(video_source)

if __name__ == "__main__":
    main()