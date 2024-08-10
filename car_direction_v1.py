import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import sys
import contextlib

# Create the frames directory if it doesn't exist
frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

# Load the video
video_path = "car_moving3.mp4"
cap = cv2.VideoCapture(video_path)

# Parameters
fps = 5
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
frame_count = 0

# Tracking variables
car_positions = deque(maxlen=2)
directions = []

def get_direction(prev_bbox, curr_bbox):
    if prev_bbox.size == 0 or curr_bbox.size == 0:
        return None
    prev_x = (prev_bbox[0] + prev_bbox[2]) / 2
    curr_x = (curr_bbox[0] + curr_bbox[2]) / 2
    if curr_x > prev_x:
        return "left to right"
    elif curr_x < prev_x:
        return "right to left"
    return None

def draw_bbox(frame, bbox):
    # Draw a rectangle around the detected object
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 255, 0)  # Green color for the bounding box
    thickness = 2
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (640, 640))

        # Perform object detection
        with suppress_stdout():
            results = model(resized_frame)

        # Draw bounding boxes and save the frame
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Filter for cars (class ID 2 in COCO dataset for 'car')
                    if int(box.cls[0]) == 2:
                        bbox = box.xyxy[0].cpu().numpy()
                        resized_frame = draw_bbox(resized_frame, bbox)
                        car_positions.append(bbox)

                        if len(car_positions) == 2:
                            direction = get_direction(car_positions[0], car_positions[1])
                            if direction:
                                directions.append(direction)

        # Save the frame with bounding boxes
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, resized_frame)

    frame_count += 1

cap.release()

# Determine the most frequent direction
if directions:
    most_common_direction = Counter(directions).most_common(1)[0][0]
    print(f"Car is moving from {most_common_direction}")
else:
    print("No car movement detected")

# Optional: Inspect saved frames to diagnose detection issues
print("Processed frames saved in the 'frames' directory for inspection.")
