# fall_detector.py
import cv2
import mediapipe as mp
import time
from config import frame_size

class FallDetector:
    def __init__(self):
        print("Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Fall detection state
        self.fall_detected = False
        self.last_hip_y = 0
        self.fall_threshold = 100
        self.fall_time_start = 0
        self.time_in_horizontal_pos = 0

        # FPS calculation state
        self.prev_time = 0

    def _get_landmark_coords(self, landmarks, landmark_name):
        """Internal helper to get landmark x, y."""
        if landmarks:
            try:
                landmark = landmarks.landmark[getattr(self.mp_pose.PoseLandmark, landmark_name).value]
                return int(landmark.x * frame_size[0]), int(landmark.y * frame_size[1]) 
            except:
                return None, None
        return None, None

    def _check_fall(self, landmarks):
        """Internal helper with fall detection logic."""
        if landmarks is None:
            return None

        left_hip_x, left_hip_y = self._get_landmark_coords(landmarks, 'LEFT_HIP')
        right_hip_x, right_hip_y = self._get_landmark_coords(landmarks, 'RIGHT_HIP')
        left_shoulder_x, left_shoulder_y = self._get_landmark_coords(landmarks, 'LEFT_SHOULDER')
        right_shoulder_x, right_shoulder_y = self._get_landmark_coords(landmarks, 'RIGHT_SHOULDER')

        if left_hip_y is None or right_hip_y is None:
            return None

        if left_shoulder_y is None or right_shoulder_y is None:
            return None

        current_hip_y = (left_hip_y + right_hip_y) / 2

        current_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

        if self.last_hip_y > 0 and (self.last_hip_y - current_hip_y) > self.fall_threshold:
            if not self.fall_detected:
                print("Potential Fall: Rapid drop detected!")
                self.fall_detected = True
                self.fall_time_start = time.time()

        hip_shoulder_height_diff = abs(current_hip_y - current_shoulder_y)
        if hip_shoulder_height_diff < 50:
            self.time_in_horizontal_pos += 1
        else:
            self.time_in_horizontal_pos = 0
        
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

    def process_frame(self, frame):
        """
        Processes a single frame:
        1. Runs pose detection.
        2. Checks for a fall.
        3. Draws landmarks and FPS.
        Returns the processed frame.
        """
        # --- FPS Timer Start ---
        current_time = time.time()
        
        # Process with MediaPipe
        frame.flags.writeable = False
        results = self.pose.process(frame)
        frame.flags.writeable = True

        # Check for fall
        fall_status = self._check_fall(results.pose_landmarks)
        if fall_status:
            cv2.putText(frame, fall_status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 0, 255), 4, cv2.LINE_AA)

        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # --- FPS Calculation and Display ---
        elapsed = current_time - self.prev_time
        fps = 0
        if elapsed > 0:
            fps = 1 / elapsed
        self.prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        # -------------------------------------
        
        return frame

    def close(self):
        """Cleans up the MediaPipe pose object."""
        self.pose.close()
        print("MediaPipe Pose closed.")