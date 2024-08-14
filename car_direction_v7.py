import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque

frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

wrong_side_dir = os.path.join(frames_dir, 'wrong_side_cars')
os.makedirs(wrong_side_dir, exist_ok=True)

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

# Load the video
video_path = "car_moving2.mp4"  # Change to your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'. Please check the file path.")
    exit()

# Parameters
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = int(fps / 5)  # Process every 5 FPS
frame_count = 0

# Tracking variables
car_tracks = {}
car_id_counter = 0
wrong_side_cars = set()
confidence_threshold = 0.4
min_frames_for_decision = 15
movement_threshold = 10
stable_threshold = 5

def get_direction(prev_bbox, curr_bbox):
    if prev_bbox.size == 0 or curr_bbox.size == 0:
        return None

    prev_x_center = (prev_bbox[0] + prev_bbox[2]) / 2
    curr_x_center = (curr_bbox[0] + curr_bbox[2]) / 2
    prev_y_center = (prev_bbox[1] + prev_bbox[3]) / 2
    curr_y_center = (curr_bbox[1] + curr_bbox[3]) / 2

    movement_x = curr_x_center - prev_x_center
    movement_y = curr_y_center - prev_y_center

    # Determine the general direction based on movement
    if abs(movement_x) > movement_threshold or abs(movement_y) > movement_threshold:
        if movement_x < 0 and movement_y < 0:
            return "wrong"
        else:
            return "correct"
    return "stable"

def draw_bbox(frame, bbox, confidence, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if color == (0, 0, 255):  # Only show confidence for wrong direction
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

        detected_car_ids = set()
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    if confidence < confidence_threshold:
                        continue

                    if int(box.cls[0]) == 2:  # Class ID for 'car'
                        bbox = box.xyxy[0].cpu().numpy()

                        matched = False
                        for car_id, track in car_tracks.items():
                            if len(track) > 0 and np.allclose(track[-1], bbox, atol=20):
                                car_tracks[car_id].append(bbox)
                                detected_car_ids.add(car_id)
                                matched = True
                                break

                        if not matched:
                            car_id_counter += 1
                            car_tracks[car_id_counter] = deque([bbox], maxlen=min_frames_for_decision)
                            detected_car_ids.add(car_id_counter)

                        for car_id in detected_car_ids:
                            if len(car_tracks[car_id]) >= min_frames_for_decision:
                                direction = get_direction(car_tracks[car_id][0], car_tracks[car_id][-1])
                                print(f"Car ID: {car_id}, Direction: {direction}")

                                if direction == "stable":
                                    continue

                                if direction == "wrong":
                                    if car_id not in wrong_side_cars:
                                        wrong_side_cars.add(car_id)
                                        resized_frame = draw_bbox(resized_frame, bbox, confidence, color=(0, 0, 255))
                                        wrong_frame_filename = os.path.join(wrong_side_dir, f"wrong_frame_{frame_count}.jpg")
                                        cv2.imwrite(wrong_frame_filename, resized_frame)
                                else:
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
