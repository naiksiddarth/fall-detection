# main.py
import cv2
import time
import threading

# Import our new modules
import shared_state
from camera import CameraStream
from fall_detector import FallDetector
import web_server


def detection_loop():
    """
    Main processing loop.
    - Gets frames from camera
    - Processes frames with fall detector
    - Handles recording logic
    - Updates streaming frame
    """
    cam = None
    detector = None
    
    try:
        # Initialize components
        cam = CameraStream()
        detector = FallDetector()
        cam.start()

        # Give camera time to warm up
        time.sleep(1.0) 
        
        while shared_state.running:
            # 1. Get Frame
            frame = cam.capture_frame()
            if frame is None:
                continue


            # 2. Process Frame
            processed_frame = detector.process_frame(frame)

            # 3. Handle Recording
            with shared_state.recording_lock:
                # Check if state *changed* from False to True (START)
                if shared_state.is_recording and shared_state.video_writer is None:
                    print("Detection loop: Starting recording...")
                    filename = f"recordings/recording_{time.strftime('%Y%m%d_%H%M%S')}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    shared_state.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (shared_state.frame_size[0], shared_state.frame_size[1]))
                    print(f"Recording to {filename}")

                # Check if state *changed* from True to False (STOP)
                elif not shared_state.is_recording and shared_state.video_writer is not None:
                    print("Detection loop: Stopping recording...")
                    shared_state.video_writer.release()
                    shared_state.video_writer = None

                # If recording is on, write the frame
                if shared_state.is_recording and shared_state.video_writer is not None:
                    # Convert to BGR for VideoWriter
                    bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    shared_state.video_writer.write(bgr_frame)

            # 4. Update Streaming Frame
            (flag, encoded_image) = cv2.imencode(".jpg", processed_frame)
            if flag:
                with shared_state.frame_lock:
                    shared_state.jpeg_frame = bytearray(encoded_image)
                    shared_state.frame_count[0] += 1
    except KeyboardInterrupt:
        print("Detection loop interrupted.")
    except Exception as e:
        print(f"\n!!! DETECTION LOOP CRASHED !!!")
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")
        shared_state.running = False # Tell web server to shut down if it's still up
        
        if cam:
            cam.stop()
        if detector:
            detector.close()
            
        with shared_state.recording_lock:
            if shared_state.video_writer is not None:
                shared_state.video_writer.release()
                print("Recording stopped on cleanup.")

if __name__ == '__main__':
    try:
        # Start the detection loop in a separate thread
        print("Starting detection thread...")
        det_thread = threading.Thread(target=detection_loop, daemon=True)
        det_thread.start()

        # Start the web server (this will block the main thread)
        web_server.start_server()

    except KeyboardInterrupt:
        print("\nCtrl+C pressed in main. Shutting down...")
        shared_state.running = False
        det_thread.join(timeout=2.0) # Wait for detection thread to finish
    finally:
        print("Script finished.")