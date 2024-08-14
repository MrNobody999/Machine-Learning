import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque

frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

model = YOLO("yolov8s.pt")

# Load the video
video_path = "car_moving3.mp4"  
cap = cv2.VideoCapture(video_path)

# Check if video file was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'. Please check the file path.")
    exit()

# Parameters
fps = 5
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
frame_count = 0

# Tracking variables
car_tracks = {}
car_id_counter = 0
wrong_side_cars = set()
car_directions = {}
wrong_side_car_ids = set()

# Direction determination thresholds
min_frames_for_decision = 15
movement_threshold = 10  # Minimum movement to be considered moving
stable_threshold = 5  # Frames for considering a vehicle stable
confidence_threshold = 0.4  # Confidence threshold

def get_direction(prev_bbox, curr_bbox):
    if prev_bbox.size == 0 or curr_bbox.size == 0:
        return None

    prev_x = (prev_bbox[0] + prev_bbox[2]) / 2
    curr_x = (curr_bbox[0] + curr_bbox[2]) / 2
    prev_y = (prev_bbox[1] + prev_bbox[3]) / 2
    curr_y = (curr_bbox[1] + curr_bbox[3]) / 2

    movement_x = curr_x - prev_x
    movement_y = curr_y - prev_y

    if abs(movement_x) > movement_threshold and abs(movement_y) > movement_threshold:
        if movement_x > 0 and movement_y > 0:
            return "lower left to upper right"
        elif movement_x > 0 and movement_y < 0:
            return "upper left to lower right"
        elif movement_x < 0 and movement_y > 0:
            return "lower right to upper left"
        elif movement_x < 0 and movement_y < 0:
            return "upper right to lower left"
        elif movement_x > 0:
            return "left to right"
        else:
            return "right to left"
    elif abs(movement_x) > movement_threshold:
        return "left to right" if movement_x > 0 else "right to left"
    elif abs(movement_y) > movement_threshold:
        return "top to bottom" if movement_y > 0 else "bottom to top"
    else:
        return "stable"

def draw_bbox(frame, bbox, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    thickness = 2
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

def is_vehicle_near(bbox, frame_shape):
    _, y1, _, y2 = bbox
    frame_height = frame_shape[0]
    vehicle_height = y2 - y1
    return vehicle_height > frame_height * 0.4  # Example threshold; adjust as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        resized_frame = cv2.resize(frame, (640, 640))

        # Perform object detection
        results = model(resized_frame)

        current_frame_tracks = {}

        # Process detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    if confidence < confidence_threshold:
                        continue  # Skip detections with confidence below threshold

                    if int(box.cls[0]) == 2:  # Class ID for 'car'
                        bbox = box.xyxy[0].cpu().numpy()

                        matched = False
                        for car_id, track in car_tracks.items():
                            if len(track) > 0 and np.allclose(track[-1], bbox, atol=20):
                                track.append(bbox)
                                matched = True
                                current_frame_tracks[car_id] = track
                                break

                        if not matched:
                            car_id_counter += 1
                            car_tracks[car_id_counter] = deque([bbox], maxlen=min_frames_for_decision)
                            current_frame_tracks[car_id_counter] = car_tracks[car_id_counter]
                            car_directions[car_id_counter] = None

        for car_id, track in current_frame_tracks.items():
            if len(track) == min_frames_for_decision:
                direction = get_direction(track[0], track[-1])
                if direction in ["right to left", "upper right to lower left", "top to bottom"]:
                    if not is_vehicle_near(track[-1], resized_frame.shape):
                        wrong_side_car_ids.add(car_id)
                    resized_frame = draw_bbox(resized_frame, track[-1], color=(0, 0, 255))
                else:
                    resized_frame = draw_bbox(resized_frame, track[-1], color=(0, 255, 0))
                car_directions[car_id] = direction
            else:
                direction = get_direction(track[0], track[-1])
                if direction in ["right to left", "upper right to lower left", "top to bottom"]:
                    if not is_vehicle_near(track[-1], resized_frame.shape):
                        wrong_side_car_ids.add(car_id)
                    resized_frame = draw_bbox(resized_frame, track[-1], color=(0, 0, 255))
                else:
                    resized_frame = draw_bbox(resized_frame, track[-1], color=(0, 255, 0))

        # Save every frame regardless of wrong detection
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, resized_frame)

    frame_count += 1

cap.release()

# Print results
print(f"Wrong side detected for {len(wrong_side_car_ids)} car(s).")

print("Processed frames saved in the 'frames' directory for inspection.")
