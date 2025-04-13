import cv2
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import json

def process_frame(frame, text_prompts, model, processor, device):
    """Process a video frame to detect objects using OWL-ViT."""
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prepare inputs for the model
        inputs = processor(text=text_prompts, images=frame_rgb, return_tensors="pt").to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor([frame.shape[:2]], device=device)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.05
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        print(f"Found {len(boxes)} potential detections")
        if len(boxes) > 0:
            print(f"Scores: {scores}")
        return boxes, scores, labels

    except Exception as e:
        print(f"Error in process_frame: {e}")
        return [], [], []

def draw_boxes(frame, boxes, scores, labels, text_prompts, threshold=0.05):
    """Draw bounding boxes with labels and scores on the frame."""
    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            box = [int(i) for i in box]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            text = f"{text_prompts[label]}: {score:.2f}"
            cv2.putText(frame, text, (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Detection: {text} at {box}")
    return frame

def log_detections(frame_id, boxes, scores, labels, text_prompts, threshold=0.05):
    """Log detections to a JSON file."""
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            detections.append({
                "label": text_prompts[label],
                "score": float(score),
                "box": [float(i) for i in box]
            })
    with open("detections.json", "a") as f:
        json.dump({"frame": frame_id, "detections": detections}, f)
        f.write("\n")