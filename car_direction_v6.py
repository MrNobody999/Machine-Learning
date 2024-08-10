import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque

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
wrong_side_cars = set()
wrong_side_counts = {}

# Direction determination threshold
min_frames_for_decision = 10

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

def draw_bbox(frame, bbox, confidence, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    thickness = 2
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"Conf: {confidence:.2f}"
    frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        resized_frame = cv2.resize(frame, (640, 640))

        # Perform object detection
        results = model(resized_frame)

        # Process detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 2:  # Class ID for 'car'
                        bbox = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])

                        matched = False
                        for car_id, track in car_tracks.items():
                            if len(track) > 0 and np.allclose(track[-1], bbox, atol=20):
                                car_tracks[car_id].append(bbox)
                                matched = True
                                break

                        if not matched:
                            car_id_counter += 1
                            car_tracks[car_id_counter] = deque([bbox], maxlen=min_frames_for_decision)

                        car_id = max(car_tracks, key=lambda k: len(car_tracks[k]))

                        if len(car_tracks[car_id]) == min_frames_for_decision:
                            direction = get_direction(car_tracks[car_id][0], car_tracks[car_id][-1])

                            if direction == "right to left":
                                wrong_side_counts[car_id] = wrong_side_counts.get(car_id, 0) + 1

                                if wrong_side_counts[car_id] >= 10:
                                    wrong_side_cars.add(car_id)
                                    resized_frame = draw_bbox(resized_frame, bbox, confidence, color=(0, 0, 255))
                                    frame_filename = os.path.join(frames_dir, f"wrong_frame_{frame_count}.jpg")
                                    cv2.imwrite(frame_filename, resized_frame)
                            else:
                                wrong_side_counts[car_id] = 0
                        else:
                            direction = get_direction(car_tracks[car_id][0], car_tracks[car_id][-1])
                            if direction == "left to right":
                                resized_frame = draw_bbox(resized_frame, bbox, confidence, color=(0, 255, 0))

        # Save every frame regardless of wrong detection
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, resized_frame)

    frame_count += 1

cap.release()

if wrong_side_cars:
    print(f"Wrong side detected for {len(wrong_side_cars)} car(s).")
else:
    print("All cars moved in the correct direction.")

print("Processed frames saved in the 'frames' directory for inspection.")
