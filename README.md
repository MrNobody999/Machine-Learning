# Machine-Learning
Wrong Direction Vehicle Detection Using YOLOv8
This project implements a vehicle detection system using YOLOv8, aimed at identifying and counting vehicles moving in the wrong direction on a roadway. The system processes video footage, identifies cars, and determines whether they are moving against the allowed traffic flow based on predefined direction rules. The solution is designed to work robustly across various scenarios, including detecting vehicles moving from different angles, avoiding false positives from parked or stable vehicles, and ensuring that each vehicle is counted only once when moving in the wrong direction.

Key Features
YOLOv8 for Object Detection: Utilizes the YOLOv8 model for high-accuracy real-time vehicle detection.
Direction Classification: Classifies vehicle movement based on bounding box positions, identifying if vehicles move in incorrect directions (e.g., right to left, upper right to lower left).
Stable and Parked Vehicle Filtering: Implements logic to filter out stationary vehicles and avoid counting parked cars as wrong-side vehicles.
Robust Vehicle Tracking: Tracks vehicles over multiple frames to ensure consistent and accurate direction classification and counting.
Custom Alerts and Counts: Generates alerts for vehicles moving in the wrong direction and provides a final count of such incidents.
Project Structure
car_direction_detection.py: Main script that handles video input, YOLOv8 detection, direction analysis, and output generation.
frames/: Directory where processed video frames are saved, showing detections and direction annotations.
YOLOv8 Model: Pretrained YOLOv8 model (yolov8s.pt) used for object detection.
Videos: Example video files used to test the system, demonstrating various scenarios including correct and incorrect vehicle movements.
How to Use
Clone the Repository:
bash
Copy code
git clone https://github.com/MrNobody999/wrong-direction-vehicle-detection.git
Install Dependencies:
Ensure you have Python and the required libraries installed:
bash
Copy code
pip install -r requirements.txt
Run the Detection:
Place your video files in the root directory and run the detection script:
bash
Copy code
python car_direction_detection.py --video_path your_video.mp4
Review Results:
Processed frames and the summary of wrong direction detections will be saved in the frames/ directory.
Future Work
Enhance the model to better handle occlusions and complex traffic scenarios.
Integrate real-time processing capabilities with streaming video input.
Improve detection accuracy for edge cases, such as heavy traffic or varying weather conditions.
License
This project is licensed under the MIT License - see the LICENSE file for details.
