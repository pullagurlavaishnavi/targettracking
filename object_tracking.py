import cv2
import numpy as np
from typing import Tuple, Optional, Dict

class ObjectTracker:
    def __init__(self):
        """
        Initialize the object tracker with KCF tracker and Kalman Filter
        """
        self.tracker = cv2.TrackerKCF_create()
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.is_tracking = False
        self.last_bbox = None

    def initialize(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Initialize tracking with a bounding box
        """
        self.is_tracking = self.tracker.init(frame, bbox)
        if self.is_tracking:
            self.last_bbox = bbox
            # Initialize Kalman Filter with the center of the bounding box
            center_x = bbox[0] + bbox[2] // 2
            center_y = bbox[1] + bbox[3] // 2
            self.kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
        return self.is_tracking

    def update(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Update tracking for the current frame
        Returns tracking results or None if tracking failed
        """
        if not self.is_tracking:
            return None

        # Update KCF tracker
        success, bbox = self.tracker.update(frame)
        
        if not success:
            self.is_tracking = False
            return None

        # Update Kalman Filter
        center_x = bbox[0] + bbox[2] // 2
        center_y = bbox[1] + bbox[3] // 2
        
        # Prediction
        prediction = self.kalman.predict()
        
        # Update with measurement
        measurement = np.array([[center_x], [center_y]], np.float32)
        self.kalman.correct(measurement)
        
        # Get filtered position
        filtered_pos = self.kalman.statePost[:2].flatten()
        
        # Calculate velocity
        velocity = self.kalman.statePost[2:].flatten()
        
        # Update last bounding box
        self.last_bbox = bbox

        return {
            "bbox": bbox,
            "center": (int(filtered_pos[0]), int(filtered_pos[1])),
            "velocity": velocity,
            "confidence": 1.0  # KCF doesn't provide confidence, could be enhanced
        }

    def reset(self):
        """
        Reset the tracker
        """
        self.is_tracking = False
        self.last_bbox = None
        self.kalman.statePre = np.array([[0], [0], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[0], [0], [0], [0]], np.float32) 