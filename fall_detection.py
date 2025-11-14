import cv2
import time
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2
from flask import Flask, Response, render_template, request, redirect, url_for
import threading
import io
import os       # --- NEW ---
import signal   # --- NEW ---

import utils

# --- Flask App Setup ---
app = Flask(__name__)

# --- Global Variables ---
# For streaming
output_frame = None
frame_lock = threading.Lock()

# --- NEW --- (For recording and program control)
running = True              # Controls the detection thread loop
is_recording = False
video_writer = None
recording_lock = threading.Lock() # Lock for is_recording and video_writer
det_thread = None           # To store the detection thread object
# -------------

# --- MediaPipe and Fall Detection Setup ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables for fall detection
fall_detected = False
last_hip_y = 0
fall_threshold = 100  # Adjust this threshold based on your camera setup (in pixels)
fall_time_start = 0
time_in_horizontal_pos = 0

def get_landmark_coords(landmarks, landmark_name):
    """Get the x, y coordinates of a specific landmark."""
    if landmarks:
        try:
            landmark = landmarks.landmark[getattr(mp_pose.PoseLandmark, landmark_name).value]
            # Convert normalized coordinates to pixel coordinates
            return int(landmark.x * 640), int(landmark.y * 480)
        except:
            return None, None
    return None, None

def check_fall(landmarks):
    """Simple fall detection logic based on hip position and orientation."""
    global last_hip_y, fall_detected, fall_time_start, time_in_horizontal_pos

    if landmarks:
        # Get coordinates for left and right hip
        left_hip_x, left_hip_y = get_landmark_coords(landmarks, 'LEFT_HIP')
        right_hip_x, right_hip_y = get_landmark_coords(landmarks, 'RIGHT_HIP')
        
        # Get coordinates for shoulders (to check orientation)
        left_shoulder_x, left_shoulder_y = get_landmark_coords(landmarks, 'LEFT_SHOULDER')
        right_shoulder_x, right_shoulder_y = get_landmark_coords(landmarks, 'RIGHT_SHOULDER')

        if left_hip_y is not None and right_hip_y is not None:
            # Average hip position
            current_hip_y = (left_hip_y + right_hip_y) / 2
            
            # --- 1. Check for rapid vertical drop ---
            if last_hip_y > 0 and (last_hip_y - current_hip_y) > fall_threshold:
                if not fall_detected:
                    print("Potential Fall: Rapid drop detected!")
                    fall_detected = True
                    fall_time_start = time.time() # Start timer

            # --- 2. Check if person is horizontal (post-fall) ---
            if left_shoulder_y is not None and right_shoulder_y is not None:
                hip_shoulder_height_diff = abs(current_hip_y - (left_shoulder_y + right_shoulder_y) / 2)
                
                # If the height difference between hips and shoulders is small,
                # the person is likely horizontal.
                if hip_shoulder_height_diff < 50: # Adjust this pixel threshold
                    time_in_horizontal_pos += 1
                else:
                    time_in_horizontal_pos = 0
            
            # --- 3. Confirm Fall ---
            if fall_detected or time_in_horizontal_pos > 60: # (60 frames ≈ 2 seconds)
                if fall_detected == False:
                        print("Potential Fall: Person is horizontal!")
                        fall_detected = True
                        fall_time_start = time.time()

                if time.time() - fall_time_start > 2.0: # Check if 2 seconds have passed
                    return "FALL DETECTED"
            else:
                fall_detected = False
                fall_time_start = 0

            last_hip_y = current_hip_y
    
    return None # No fall

def detection_thread():
    """Main thread for fall detection and image processing."""
    # --- MODIFIED ---
    global output_frame, frame_lock, running
    global is_recording, video_writer, recording_lock
    
    print("Starting camera feed...")


    # --- FPS Calculation Variables ---
    prev_time = 0 
    fps = 0       
    # ---------------------------------

    try:
        # --- MODIFIED --- (Loop now checks the 'running' flag)
        while running:
            # --- FPS Timer Start ---
            current_time = time.time() 
            # -----------------------

            # Capture a frame
            im = utils.get_current_frame()
            
            # Process the image with MediaPipe
            im.flags.writeable = False
            results = pose.process(im)
            im.flags.writeable = True

            # Check for fall
            fall_status = check_fall(results.pose_landmarks)
            
            if fall_status:
                # Display Fall Alert
                cv2.putText(im, fall_status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, (0, 0, 255), 4, cv2.LINE_AA)

            # Draw the pose annotation on the image.
            mp_drawing.draw_landmarks(
                im,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # --- FPS Calculation and Display ---
            elapsed = current_time - prev_time 
            if elapsed > 0:                    
                fps = 1 / elapsed              
            prev_time = current_time           
            
            # Draw FPS on the image
            cv2.putText(im, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA) 
            # -------------------------------------

            # --- NEW --- (Recording Logic)
            with recording_lock:
                if is_recording and video_writer is not None:
                    # Convert RGB (from PiCamera/MediaPipe) to BGR (for OpenCV VideoWriter)
                    bgr_frame = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                    video_writer.write(bgr_frame)
            # -----------

            # Update the global frame for streaming
            with frame_lock:
                output_frame = im.copy()
                
    except KeyboardInterrupt:
        print("Detection thread caught interrupt...")
    finally:
        # --- MODIFIED --- (Added cleanup)
        running = False # Ensure loop condition is false
        
        # Clean up recording
        with recording_lock:
            if video_writer is not None:
                video_writer.release()
                print("Recording stopped on cleanup.")
        
        # Clean up MediaPipe and Camera
        pose.close()
        print("Detection thread cleaned up.")

def generate_frames():
    """Generator function to stream JPEG frames."""
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                # Send a placeholder or wait
                time.sleep(0.1)
                continue
            
            # Encode the frame as JPEG
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        
        # Yield the frame in the multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- MODIFIED --- (Updated HTML with buttons)
@app.route('/')
def index():
    """Simple HTML page to display the video stream and controls."""
    global is_recording, recording_lock
    
    with recording_lock:
        rec_status = is_recording
    
    if rec_status:
        record_button_text = "Stop Recording"
    else:
        record_button_text = "Start Recording"

    # Using f-string for dynamic HTML
    return f"""
    <html>
    <head><title>Fall Detection Stream</title></head>
    <body>
        <h1>Fall Detection Stream</h1>
        <p>Live feed from Raspberry Pi (Port {2131})</p>
        <img src="/video_feed" width="640" height="480">
        
        <hr>
        
        <form action="/toggle_record" method="post" style="display:inline-block;">
            <input type="submit" value="{record_button_text}" style="padding: 10px; font-size: 16px;">
        </form>
        
        <form action="/shutdown" method="post" style="display:inline-block; margin-left: 20px;">
            <input type="submit" value="Exit Program" style="padding: 10px; font-size: 16px; background-color: #ffcccc;">
        </form>
    </body>
    </html>
    """

# --- NEW --- (Route to handle recording)
@app.route('/toggle_record', methods=['POST'])
def toggle_record():
    global is_recording, video_writer, recording_lock
    
    with recording_lock:
        if is_recording:
            # --- STOP RECORDING ---
            print("Stopping recording...")
            is_recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            print("Recording stopped.")
        else:
            # --- START RECORDING ---
            print("Starting recording...")
            is_recording = True
            # Create a unique filename with timestamp
            filename = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.avi"
            
            # Get video properties
            frame_size = (640, 480) # Must match camera config
            output_fps = 20.0       # Set a reasonable output FPS
            
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # .avi codec
            video_writer = cv2.VideoWriter(filename, fourcc, output_fps, frame_size)
            print(f"Recording started. Saving to {filename}")
    
    # Redirect back to the index page to update the button text
    return redirect(url_for('index'))

# --- NEW --- (Route to exit the program)
@app.route('/shutdown', methods=['POST'])
def shutdown():
    global running
    print("Shutdown request received via web...")
    
    # Stop the detection thread
    running = False 
    
    # Give the thread time to clean up
    time.sleep(1.0)
    
    # Stop the Flask server by sending SIGINT (Ctrl+C) to our own process
    os.kill(os.getpid(), signal.SIGINT)
    return "Server is shutting down..."


if __name__ == '__main__':
    try:
        # --- MODIFIED ---
        
        # Start the fall detection in a separate thread
        print("Starting detection thread...")
        det_thread = threading.Thread(target=detection_thread)
        det_thread.daemon = True
        det_thread.start()

        # Start the Flask web server
        # Host '0.0.0.0' makes it accessible on your network
        print(f"Starting web server on port {2131}...")
        app.run(host='0.0.0.0', port=2131, threaded=True)

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Shutting down server and cleaning up...")
        running = False # Signal thread to stop
        if det_thread is not None:
            det_thread.join() # Wait for thread to finish
    finally:
        print("Script finished.")
