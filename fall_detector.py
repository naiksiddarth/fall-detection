import cv2
import mediapipe as mp
import time
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import shared_state


from collections import deque



class FallDetector:
    def __init__(self, model_path='pose_landmarker_lite.task'):
        print("Initializing MediaPipe Tasks Pose Landmarker...")

        # --- FIX: Suppress MediaPipe/TensorFlow Console Spam ---
        os.environ['GLOG_minloglevel'] = '2'

        self.frame_buffer = deque(maxlen=150)
        self.cord_buffer = deque(maxlen=30)

        # --- FIX: Verify file existence ---
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")

        # --- FIX: Load model into memory (Buffer) ---
        with open(model_path, 'rb') as f:
            model_buffer = f.read()
        
        # 1. Create BaseOptions using the BUFFER
        base_options = python.BaseOptions(model_asset_buffer=model_buffer)
        
        # 2. Create PoseLandmarkerOptions
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.85,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.7
        )
        
        # 3. Create the Landmarker
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # Fall detection state
        self.fall_detected = False
        self.fall_threshold = 100
        self.time_in_horizontal_pos = 0

        # FPS calculation state
        self.prev_time = 0
        
        # Manual frame timestamp counter
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

        if pose_landmarks is None:
            self.cord_buffer.clear()
            self.frame_buffer.clear()
            return None
        
        if shared_state.fall_detected:
            return None

        left_hip_x, left_hip_y = self._get_landmark_coords(pose_landmarks, 23)
        right_hip_x, right_hip_y = self._get_landmark_coords(pose_landmarks, 24)
        left_shoulder_x, left_shoulder_y = self._get_landmark_coords(pose_landmarks, 11)
        right_shoulder_x, right_shoulder_y = self._get_landmark_coords(pose_landmarks, 12)
        if left_hip_y is None or right_hip_y is  None:
            return None

        if left_shoulder_y is None or right_shoulder_y is None:
            return None

        current_hip_y = (left_hip_y + right_hip_y) / 2
                
        if not shared_state.check_rapid_fall:
            return None

        try :
            start_cord = self.cord_buffer[0]
            end_cord = self.cord_buffer[-1]
            fall_speed = (end_cord - start_cord) 
            if fall_speed >= self.fall_threshold:
                shared_state.rapid_fall = True
        except IndexError:
            pass
        if shared_state.rapid_fall:
            hip_shoulder_height_diff = abs(current_hip_y - (left_shoulder_y + right_shoulder_y) / 2)
            shared_state.hip_shoulder_height_diff = hip_shoulder_height_diff
            print(hip_shoulder_height_diff)
            if hip_shoulder_height_diff < 50:
                self.time_in_horizontal_pos += 1
            else:
                self.time_in_horizontal_pos = 0

            if self.time_in_horizontal_pos > 60:
                shared_state.fall_detected = True
                return "FALL DETECTED"

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
        Processes a single frame.
        """
        # --- FPS Timer Start ---
        current_time = time.time()
        
        # --- FIX: Ensure memory is contiguous ---
        # Picamera2 on Raspberry Pi often returns padded arrays (strides).
        # MediaPipe expects packed arrays. This fixes "scrambled" or "rubbish" video.
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
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
        cv2.putText(frame, f"Height diff: {shared_state.hip_shoulder_height_diff:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        
        return frame

    def close(self):
        """Cleans up the Landmarker."""
        self.landmarker.close()
        print("MediaPipe Tasks Pose Landmarker closed.")