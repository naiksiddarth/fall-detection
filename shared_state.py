# shared_state.py
import threading

# --- Threading & Control ---
running = True              # Flag to control the main detection loop
frame_lock = threading.Lock() # Lock for accessing the jpeg_frame
recording_lock = threading.Lock() # Lock for accessing recording state

# --- Streaming ---
jpeg_frame = None           # Stores the latest *encoded* JPEG frame for streaming
frame_count = [0]
# --- Recording ---
is_recording = False
video_writer = None

# how to check fall
check_rapid_fall = False



hip_shoulder_height_diff = 0

