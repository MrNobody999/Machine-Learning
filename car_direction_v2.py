import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import sys
import contextlib

frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

# Load the video
video_path = "car_moving.mp4"
cap = cv2.VideoCapture(video_path)

# Parameters
fps = 5
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
frame_count = 0

# Tracking variables
car_positions = deque(maxlen=2)
alerts = set()  # To store unique alerts for each car

# Correct horizontal and vertical directions
correct_horizontal_direction = "left to right"
correct_vertical_direction = "downside to upside"

def get_direction(prev_bbox, curr_bbox):
    if prev_bbox.size == 0 or curr_bbox.size == 0:
        return None, None
    prev_x = (prev_bbox[0] + prev_bbox[2]) / 2
    curr_x = (curr_bbox[0] + curr_bbox[2]) / 2
    prev_y = (prev_bbox[1] + prev_bbox[3]) / 2
    curr_y = (curr_bbox[1] + curr_bbox[3]) / 2
    
    if curr_x > prev_x:
        horizontal_direction = "left to right"
    elif curr_x < prev_x:
        horizontal_direction = "right to left"
    else:
        horizontal_direction = None
    
    if curr_y > prev_y:
        vertical_direction = "upside to downside"
    elif curr_y < prev_y:
        vertical_direction = "downside to upside"
    else:
        vertical_direction = None
    
    return horizontal_direction, vertical_direction

def draw_bbox(frame, bbox, color=(0, 255, 0)):
    # Draw a rectangle around the detected object
    x1, y1, x2, y2 = map(int, bbox)
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
                        car_positions.append(bbox)

                        if len(car_positions) == 2:
                            horizontal_direction, vertical_direction = get_direction(car_positions[0], car_positions[1])
                            alert_color = (0, 255, 0)  # Default green color for correct direction

                            if horizontal_direction and horizontal_direction != correct_horizontal_direction:
                                alert = f"Alert: Wrong direction! Car is moving from {horizontal_direction}"
                                if alert not in alerts:
                                    alerts.add(alert)
                                    alert_color = (0, 0, 255)  # Red color for wrong direction

                            if vertical_direction and vertical_direction != correct_vertical_direction:
                                alert = f"Alert: Wrong direction! Car is moving from {vertical_direction}"
                                if alert not in alerts:
                                    alerts.add(alert)
                                    alert_color = (0, 0, 255)  # Red color for wrong direction

                            resized_frame = draw_bbox(resized_frame, bbox, color=alert_color)

        # Save the frame with bounding boxes
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, resized_frame)

    frame_count += 1

cap.release()

# Print alerts if any
if alerts:
    for alert in alerts:
        print(alert)
else:
    print("No wrong direction detected")

# Optional: Inspect saved frames to diagnose detection issues
print("Processed frames saved in the 'frames' directory for inspection.")
