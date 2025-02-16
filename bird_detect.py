"""Bird Detection System using YOLOv8

This module provides a bird detection system that uses YOLOv8 object detection model
to detect birds in video files. It processes video frames, detects birds, draws bounding
boxes around them, and saves the top detections with highest confidence scores.

The system is designed to process all MP4 files in the source_videos directory and
save detected bird images in the saved_images directory.
"""

import time
import cv2
import datetime
import os
from ultralytics import YOLO

class BirdDetector:
    """Bird detection class using YOLOv8 model
    
    Attributes:
        script_dir (str): Directory path of the script
        save_dir (str): Directory to save detected bird images
        model_path (str): Path to YOLOv8 model file
        frame_width (int): Width of processed video frames
        frame_height (int): Height of processed video frames
        detections (list): List to store detection information
        frame_count (int): Counter for processed frames
        model: YOLOv8 model instance
        cap: Video capture object
    """
    
    def __init__(self, model_path="model/yolov8n.pt", source_video="source_videos"):
        """Initialize BirdDetector with model and video source
        
        Args:
            model_path (str): Path to YOLOv8 model file
            source_video (str): Path to video file or directory
        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(self.script_dir, "saved_images")
        self.model_path = os.path.join(self.script_dir, model_path)

        self.frame_width = 1280
        self.frame_height = 720
        self.detections = []
        self.frame_count = 0
        
        # Ensure the save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Load model and video
        self.model = YOLO(model_path, "v8")
        print(f"source_video: {source_video}")

        self.cap = cv2.VideoCapture(source_video)
        
        if not self.cap.isOpened():
            print("Cannot open video stream")
            exit()

    def process_frame(self, frame):
        """Process a single video frame for bird detection
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with bird detections drawn
        """
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        detect_params = self.model.predict(source=[frame], conf=0.7, save=False)
        DP = detect_params[0].cpu().numpy()

        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]
                clsID = box.cls.cpu().numpy()[0]
                conf = box.conf.cpu().numpy()[0]
                bb = box.xyxy.cpu().numpy()[0]
                c = box.cls
                class_name = self.model.names[int(c)]

                if 'bird' in class_name.lower():
                    # Draw bounding box around detected bird
                    cv2.rectangle(
                        frame,
                        (int(bb[0]), int(bb[1])),
                        (int(bb[2]), int(bb[3])),
                        (0, 255, 0),
                        3,
                    )
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        frame,
                        class_name + " " + str(round(conf, 3)) + "%",
                        (int(bb[0]), int(bb[1]) - 10),
                        font,
                        1,
                        (255,255,255),
                        2,
                    )
                    
                    # Store detection information after drawing
                    self.detections.append({
                        'frame': self.frame_count,
                        'confidence': conf,
                        'frame_data': frame.copy(),
                        'bounding_box': bb
                    })
                    
            self.frame_count += 1
        return frame

    def save_top_detections(self):
        """Save top 3 detections with highest confidence scores"""
        if self.detections:
            self.detections.sort(key=lambda x: x['confidence'], reverse=True)
            for i, detection in enumerate(self.detections[:3]):
                current_time = datetime.datetime.now()
                filename = os.path.join(self.save_dir, 
                    f"top_{i+1}_bird_{current_time.strftime('%Y-%m-%d_%H-%M-%S-%f')}.jpg")
                cv2.imwrite(filename, detection['frame_data'])
                print(f"Top {i+1} detection saved: {filename} (Confidence: {detection['confidence']:.3f})")

    def run(self):
        """Main processing loop for video detection"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No video frame available")
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow("Object Detection", processed_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.save_top_detections()
        self.cap.release()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    # Iterate through all MP4 files in the directory
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_videos_dir = os.path.join(script_dir, "source_videos")
    for filename in os.listdir(source_videos_dir):
        if filename.lower().endswith('.mp4'):
            source_video = os.path.join(source_videos_dir, filename)
            print(f"\nProcessing: {source_video}")
            detector = BirdDetector(source_video=source_video)
            detector.run()
