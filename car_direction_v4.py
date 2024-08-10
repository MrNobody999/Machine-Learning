import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque
import sys
import contextlib

# Create the frames directory if it doesn't exist
frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

# Load the video
video_path = "car_moving.mp4"  # Change to your video file path
cap = cv2.VideoCapture(video_path)

# Parameters
fps = 5
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
frame_count = 0

# Tracking variables
car_tracks = {}
car_id_counter = 0
car_directions_alerted = {}

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

        # Process detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Filter for cars (class ID 2 in COCO dataset for 'car')
                    if int(box.cls[0]) == 2:
                        bbox = box.xyxy[0].cpu().numpy()

                        # Generate a unique ID for each car detected
                        car_id = len(car_tracks) + 1  # Simple counter for car IDs
                        if car_id not in car_tracks:
                            car_tracks[car_id] = deque(maxlen=2)

                        # Update the car's tracked positions
                        car_tracks[car_id].append(bbox)

                        # Check if we have enough positions to determine direction
                        if len(car_tracks[car_id]) == 2:
                            direction = get_direction(car_tracks[car_id][0], car_tracks[car_id][1])

                            # Check if the car is moving in the wrong direction
                            if direction == "right to left" and car_id not in car_directions_alerted:
                                print(f"Wrong side detected for car {car_id}")
                                resized_frame = draw_bbox(resized_frame, bbox, color=(0, 0, 255))  # Red color for wrong direction
                                car_directions_alerted[car_id] = True  # Alert only once per car
                            elif direction == "left to right":
                                resized_frame = draw_bbox(resized_frame, bbox)  # Green color for correct direction
                            else:
                                resized_frame = draw_bbox(resized_frame, bbox)  # Default to green if no movement is detected
                        else:
                            resized_frame = draw_bbox(resized_frame, bbox)  # Default to green if no movement is detected

        # Save the frame with bounding boxes
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, resized_frame)

    frame_count += 1

cap.release()

print("Processed frames saved in the 'frames' directory for inspection.")
