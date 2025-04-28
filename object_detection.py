import cv2
import numpy as np
from scipy.spatial import distance
from skimage import measure
from typing import Tuple, List, Dict, Optional

class ObjectDetector:
    def __init__(self, use_sift: bool = True):
        """
        Initialize the object detector with either SIFT or ORB
        Args:
            use_sift (bool): If True, use SIFT, else use ORB
        """
        self.use_sift = use_sift
        print(f"[ObjectDetector] Initializing with use_sift={use_sift}")
        if use_sift:
            print("[ObjectDetector] Creating SIFT detector...")
            self.detector = cv2.SIFT_create()
            print("[ObjectDetector] Creating BFMatcher with NORM_L2 (for SIFT ratio test)...")
            # Use NORM_L2 for SIFT; crossCheck MUST be False for knnMatch
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            print("[ObjectDetector] Creating ORB detector...")
            self.detector = cv2.ORB_create(nfeatures=1000) # Consider tuning nfeatures
            print("[ObjectDetector] Creating BFMatcher with NORM_HAMMING and crossCheck (for ORB)...")
            # Use NORM_HAMMING for ORB, crossCheck is suitable here
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        print("[ObjectDetector] Initialization complete.")

    def detect_features(self, image: np.ndarray) -> Tuple[Optional[List[cv2.KeyPoint]], Optional[np.ndarray]]:
        """
        Detect keypoints and compute descriptors in the image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between two images
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        
        if self.use_sift:
            # For SIFT, use ratio test instead of crossCheck
            # knnMatch finds k=2 nearest neighbors for each descriptor
            raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            matches = []
            for m, n in raw_matches:
                if m.distance < 0.75 * n.distance:  # Ratio test threshold
                    matches.append(m)
        else:
            # For ORB, just use regular matching with crossCheck
            matches = self.matcher.match(desc1, desc2)
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        print(f"Found {len(matches)} matches")
        
        return matches

    def find_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in the image using edge detection
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def simplify_contour(self, contour: np.ndarray, epsilon: float = 0.02) -> np.ndarray:
        """
        Simplify contour using Ramer-Douglas-Peucker algorithm
        """
        return cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)

    def detect_object(self, frame: np.ndarray, template: np.ndarray) -> Dict:
        """
        Detect object in frame using template matching
        Returns dictionary with detection results
        """
        # Detect features in both images
        kp1, desc1 = self.detect_features(template)
        kp2, desc2 = self.detect_features(frame)

        if desc1 is None or desc2 is None:
            print("DEBUG: One of the descriptor sets is None")
            return {"detected": False, "location": None, "confidence": 0.0}

        # Match features
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < 10:  # Minimum matches threshold
            print(f"DEBUG: Not enough matches: {len(matches)} < 10")
            return {"detected": False, "location": None, "confidence": 0.0}

        # Get matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography
        print(f"DEBUG: Finding homography with {len(matches)} matches")
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            print("DEBUG: Homography is None")
            return {"detected": False, "location": None, "confidence": 0.0}

        # Count inliers (matches that fit the homography)
        inliers = np.sum(mask)
        print(f"DEBUG: Homography found with {inliers}/{len(matches)} inliers")
        
        # Stronger requirement for detection - need good portion of inliers
        if inliers < 8:  # Adjust this threshold as needed
            print(f"DEBUG: Not enough inliers: {inliers} < 8")
            return {"detected": False, "location": None, "confidence": 0.0}

        # Get template corners
        h, w = template.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        # Transform corners
        try:
            dst = cv2.perspectiveTransform(pts, H)
            
            # Calculate center and confidence
            center = np.mean(dst, axis=0)[0]
            confidence = inliers / len(matches)  # Better confidence measure

            print(f"DEBUG: Object detected! Center: {center}, Confidence: {confidence:.2f}")
            
            return {
                "detected": True,
                "location": center,
                "confidence": confidence,
                "corners": dst.reshape(-1, 2)
            }
        except Exception as e:
            print(f"DEBUG: Error during perspective transform: {e}")
            return {"detected": False, "location": None, "confidence": 0.0} 