import cv2
import mediapipe as mp
import time
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FallDetector:
    def __init__(self, model_path='pose_landmarker_lite.task'):
        print("Initializing MediaPipe Tasks Pose Landmarker...")

        # --- FIX: Verify file existence to debug path issues ---
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")

        # --- FIX: Load model into memory (Buffer) ---
        # Passing the file path directly to C++ sometimes fails on Linux/Pi.
        # Reading it as bytes in Python is much more robust.
        with open(model_path, 'rb') as f:
            model_buffer = f.read()
        
        # 1. Create BaseOptions using the BUFFER, not the path
        base_options = python.BaseOptions(model_asset_buffer=model_buffer)
        
        # 2. Create PoseLandmarkerOptions
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.8,
            min_pose_presence_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # 3. Create the Landmarker
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # Fall detection state
        self.fall_detected = False
        self.last_hip_y = 0
        self.fall_threshold = 100
        self.fall_time_start = 0
        self.time_in_horizontal_pos = 0

        # FPS calculation state
        self.prev_time = 0
        
        # Use a manual frame timestamp counter
        self.current_timestamp_ms = 0

    def _get_landmark_coords(self, landmark_list, index):
        """Helper to get pixel coordinates from the new landmark object."""
        try:
            landmark = landmark_list[index]
            return int(landmark.x * 640), int(landmark.y * 480)
        except IndexError:
            return None, None

    def _check_fall(self, pose_landmarks):
        """Fall detection logic adapted for Tasks API result structure."""
        if pose_landmarks:
            # Indices: Left Hip(23), Right Hip(24), Left Shoulder(11), Right Shoulder(12)
            left_hip_x, left_hip_y = self._get_landmark_coords(pose_landmarks, 23)
            right_hip_x, right_hip_y = self._get_landmark_coords(pose_landmarks, 24)
            left_shoulder_x, left_shoulder_y = self._get_landmark_coords(pose_landmarks, 11)
            right_shoulder_x, right_shoulder_y = self._get_landmark_coords(pose_landmarks, 12)

            if left_hip_y is not None and right_hip_y is not None:
                current_hip_y = (left_hip_y + right_hip_y) / 2
                
                # 1. Rapid Drop Check
                if self.last_hip_y > 0 and (self.last_hip_y - current_hip_y) > self.fall_threshold:
                    if not self.fall_detected:
                        print("Potential Fall: Rapid drop detected!")
                        self.fall_detected = True
                        self.fall_time_start = time.time()

                # 2. Horizontal Orientation Check
                if left_shoulder_y is not None and right_shoulder_y is not None:
                    hip_shoulder_height_diff = abs(current_hip_y - (left_shoulder_y + right_shoulder_y) / 2)
                    if hip_shoulder_height_diff < 50:
                        self.time_in_horizontal_pos += 1
                    else:
                        self.time_in_horizontal_pos = 0
                
                # 3. Confirmation
                if self.fall_detected or self.time_in_horizontal_pos > 60:
                    if not self.fall_detected:
                        print("Potential Fall: Person is horizontal!")
                        self.fall_detected = True
                        self.fall_time_start = time.time()
                    if time.time() - self.fall_time_start > 2.0:
                        return "FALL DETECTED"
                else:
                    self.fall_detected = False
                    self.fall_time_start = 0
                
                self.last_hip_y = current_hip_y
        return None

    def _draw_landmarks_manually(self, frame, landmarks):
        """Custom drawing function using OpenCV."""
        if not landmarks:
            return

        # Define connections (Standard Pose topology)
        connections = mp.solutions.pose.POSE_CONNECTIONS
        
        # Draw connections (lines)
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = self._get_landmark_coords(landmarks, start_idx)
            end_point = self._get_landmark_coords(landmarks, end_idx)
            
            if start_point[0] is not None and end_point[0] is not None:
                cv2.line(frame, start_point, end_point, (245, 66, 230), 2)

        # Draw landmarks (dots)
        for i, landmark in enumerate(landmarks):
            cx, cy = int(landmark.x * 640), int(landmark.y * 480)
            cv2.circle(frame, (cx, cy), 4, (245, 117, 66), -1)

    def process_frame(self, frame):
        """
        Processes a single frame:
        1. Converts to MP Image.
        2. Runs detection (VIDEO mode) with synthetic timestamp.
        3. Checks fall.
        4. Draws visuals.
        """
        # --- FPS Timer Start ---
        current_time = time.time()
        
        # 1. Convert numpy array (frame) to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # 2. Calculate Timestamp
        self.current_timestamp_ms += 33
        
        # 3. Detect
        detection_result = self.landmarker.detect_for_video(mp_image, self.current_timestamp_ms)
        
        # 4. Extract Landmarks
        current_landmarks = None
        if detection_result.pose_landmarks:
            current_landmarks = detection_result.pose_landmarks[0] # Get first person

        # 5. Check for fall
        fall_status = self._check_fall(current_landmarks)
        if fall_status:
            cv2.putText(frame, fall_status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 0, 255), 4, cv2.LINE_AA)

        # 6. Draw landmarks
        self._draw_landmarks_manually(frame, current_landmarks)

        # --- FPS Calculation and Display ---
        elapsed = current_time - self.prev_time
        fps = 0
        if elapsed > 0:
            fps = 1 / elapsed
        self.prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame

    def close(self):
        """Cleans up the Landmarker."""
        self.landmarker.close()
        print("MediaPipe Tasks Pose Landmarker closed.")