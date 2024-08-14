import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque

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
car_positions = {}
alerts = set()

# Correct direction
correct_direction = "left to right"

def get_direction(prev_bbox, curr_bbox):
    if prev_bbox.size == 0 or curr_bbox.size == 0:
        return None
    prev_x = (prev_bbox[0] + prev_bbox[2]) / 2
    curr_x = (curr_bbox[0] + curr_bbox[2]) / 2
    
    if curr_x > prev_x:
        return "left to right"
    elif curr_x < prev_x:
        return "right to left"
    else:
        return None

def draw_bbox(frame, bbox, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    thickness = 2
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        resized_frame = cv2.resize(frame, (640, 640))

        results = model(resized_frame)

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 2:
                        bbox = box.xyxy[0].cpu().numpy()
                        car_id = int(box.id) if box.id is not None else len(car_positions)

                        if car_id not in car_positions:
                            car_positions[car_id] = deque(maxlen=2)

                        car_positions[car_id].append(bbox)

                        if len(car_positions[car_id]) == 2:
                            direction = get_direction(car_positions[car_id][0], car_positions[car_id][1])
                            alert_color = (0, 255, 0)

                            if direction and direction != correct_direction:
                                alert = f"Alert: Wrong direction! Car is moving from {direction}"
                                if car_id not in alerts:
                                    alerts.add(car_id)
                                    alert_color = (0, 0, 255)
                                    print(alert)

                            resized_frame = draw_bbox(resized_frame, bbox, color=alert_color)

        frame_filename = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, resized_frame)

    frame_count += 1

cap.release()

print("Processed frames saved in the 'frames' directory for inspection.")
