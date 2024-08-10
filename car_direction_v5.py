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
video_path = "car_moving2.mp4"  # Change to your video file path
cap = cv2.VideoCapture(video_path)

# Parameters
fps = 5
frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
frame_count = 0

# Tracking variables
car_tracks = {}
car_id_counter = 0
wrong_side_cars = set()

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (640, 640))

        # Perform object detection
        results = model(resized_frame)

        # Process detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Filter for cars (class ID 2 in COCO dataset for 'car')
                    if int(box.cls[0]) == 2:
                        bbox = box.xyxy[0].cpu().numpy()

                        # Check if the car already exists in the car_tracks
                        matched = False
                        for car_id, track in car_tracks.items():
                            if len(track) > 0 and np.allclose(track[-1], bbox, atol=20):  # Adjust the tolerance as needed
                                car_tracks[car_id].append(bbox)
                                matched = True
                                break

                        if not matched:
                            car_id_counter += 1
                            car_tracks[car_id_counter] = deque([bbox], maxlen=2)

                        car_id = max(car_tracks, key=lambda k: len(car_tracks[k]))  # Get the latest tracked car

                        # Check direction
                        if len(car_tracks[car_id]) == 2:
                            direction = get_direction(car_tracks[car_id][0], car_tracks[car_id][1])

                            if direction == "right to left":
                                wrong_side_cars.add(car_id)
                                resized_frame = draw_bbox(resized_frame, bbox, color=(0, 0, 255))  # Red color for wrong direction
                            elif direction == "left to right":
                                resized_frame = draw_bbox(resized_frame, bbox, color=(0, 255, 0))  # Green color for correct direction
                        else:
                            resized_frame = draw_bbox(resized_frame, bbox)  # Default to green if no movement is detected

        # Save the frame with bounding boxes
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, resized_frame)

    frame_count += 1

cap.release()

# Generate alert at the end of the script
if wrong_side_cars:
    print(f"Wrong side detected for {len(wrong_side_cars)} car(s).")
else:
    print("All cars moved in the correct direction.")

print("Processed frames saved in the 'frames' directory for inspection.")
