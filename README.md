# Machine-Learning
Wrong Direction Vehicle Detection Using YOLOv8
This project implements a vehicle detection system using YOLOv8, aimed at identifying and counting vehicles moving in the wrong direction on a roadway. The system processes video footage, identifies cars, and determines whether they are moving against the allowed traffic flow based on predefined direction rules. The solution is designed to work robustly across various scenarios, including detecting vehicles moving from different angles, avoiding false positives from parked or stable vehicles, and ensuring that each vehicle is counted only once when moving in the wrong direction.

Key Features
1.     YOLOv8 for Object Detection: Utilizes the YOLOv8 model for high-accuracy real-time vehicle detection.
2.     Direction Classification: Classifies vehicle movement based on bounding box positions, identifying if vehicles move in incorrect directions (e.g., right to left, upper right to lower left).
3.     Stable and Parked Vehicle Filtering: Implements logic to filter out stationary vehicles and avoid counting parked cars as wrong-side vehicles.
4.     Robust Vehicle Tracking: Tracks vehicles over multiple frames to ensure consistent and accurate direction classification and counting.
5.     Custom Alerts and Counts: Generates alerts for vehicles moving in the wrong direction and provides a final count of such incidents.


Project Structure:
1.      car_direction_detection.py: Main script that handles video input, YOLOv8 detection, direction analysis, and output generation.
2.      frames/: Directory where processed video frames are saved, showing detections and direction annotations.
3.      YOLOv8 Model: Pretrained YOLOv8 model (yolov8s.pt) used for object detection.
4.      Videos: Example video files used to test the system, demonstrating various scenarios including correct and incorrect vehicle movements.


How to Use:
1.Clone the Repository:

    git clone https://github.com/MrNobody999/wrong-direction-vehicle-detection.git


2.Install Dependencies:

    Ensure you have Python and the required libraries installed:
    
    pip install -r requirements.txt

3.Run the Detection:

    Place your video files in the root directory and run the detection script:
    
    python car_direction_detection.py --video_path your_video.mp4
  

4.Review Results:

    Processed frames and the summary of wrong direction detections will be saved in the frames/ directory.


Future Work:
1.      Enhance the model to better handle occlusions and complex traffic scenarios.
2.      Integrate real-time processing capabilities with streaming video input.
3.      Improve detection accuracy for edge cases, such as heavy traffic or varying weather conditions.

License:
      This project is licensed under the MIT License - see the LICENSE file for details.
