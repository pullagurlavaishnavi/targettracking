import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import os
from object_detection import ObjectDetector
from object_tracking import ObjectTracker

class VideoProcessor:
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the video processor
        Args:
            output_dir (str): Directory to save processed videos and clips
        """
        self.output_dir = output_dir
        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()
        self.detection_timestamps = []
        self.clip_segments = []
        os.makedirs(output_dir, exist_ok=True)

    def process_video(self, video_path: str, template_path: str) -> Dict:
        """
        Process a video file with object detection and tracking
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read template image
        template = cv2.imread(template_path)
        if template is None:
            raise ValueError(f"Could not open template image: {template_path}")

        # Prepare output video
        output_path = os.path.join(self.output_dir, f"processed_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        is_tracking = False
        current_segment = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps
            frame_count += 1

            if not is_tracking:
                # Try to detect object
                detection = self.detector.detect_object(frame, template)
                if detection["detected"] and detection["confidence"] > 0.5:
                    # Initialize tracking
                    x, y = detection["location"]
                    bbox = (int(x - template.shape[1]/2), int(y - template.shape[0]/2),
                           template.shape[1], template.shape[0])
                    is_tracking = self.tracker.initialize(frame, bbox)
                    
                    if is_tracking:
                        self.detection_timestamps.append(timestamp)
                        current_segment = {"start": timestamp, "end": None}
                        self.clip_segments.append(current_segment)

            else:
                # Update tracking
                tracking_result = self.tracker.update(frame)
                if tracking_result is None:
                    is_tracking = False
                    if current_segment:
                        current_segment["end"] = timestamp
                else:
                    # Draw tracking results
                    bbox = tracking_result["bbox"]
                    center = tracking_result["center"]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                                (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                                (0, 255, 0), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Add timestamp to frame
            cv2.putText(frame, f"Time: {timestamp:.2f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(frame)

        # Close the last segment if still open
        if current_segment and current_segment["end"] is None:
            current_segment["end"] = frame_count / fps

        cap.release()
        out.release()

        return {
            "output_path": output_path,
            "detection_timestamps": self.detection_timestamps,
            "clip_segments": self.clip_segments
        }

    def extract_clips(self, video_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Extract video clips based on detection segments
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "clips")
        os.makedirs(output_dir, exist_ok=True)

        clip_paths = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i, segment in enumerate(self.clip_segments):
            start_frame = int(segment["start"] * fps)
            end_frame = int(segment["end"] * fps)
            
            # Set video position to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Prepare output video
            clip_path = os.path.join(output_dir, f"clip_{i+1}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(clip_path, fourcc, fps,
                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            # Extract frames
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()
            clip_paths.append(clip_path)

        cap.release()
        return clip_paths 