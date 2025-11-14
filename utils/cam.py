# camera_util.py
from picamera2 import Picamera2
import threading

# Global variables for singleton pattern
_picam2_instance = None
_camera_lock = threading.Lock()
_camera_started = False


def _initialize_camera():
    """
    Internal function to initialize the Picamera2 instance only once.
    Threads are locked to avoid race conditions.
    """
    global _picam2_instance, _camera_started
    
    with _camera_lock:
        if _picam2_instance is None:
            _picam2_instance = Picamera2()
            _picam2_instance.configure(_picam2_instance.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
        
        if not _camera_started:
            _picam2_instance.start()
            _camera_started = True


def get_current_frame():
    """
    Returns the latest frame from the Raspberry Pi camera.
    Ensures the camera is initialized once and reused.
    """
    if _picam2_instance is None:
        _initialize_camera()

    return _picam2_instance.capture_array()


def shutdown_camera():
    """
    Optional cleanup function to stop the camera.
    You can call this when the program exits.
    """
    global _picam2_instance, _camera_started
    
    with _camera_lock:
        if _picam2_instance and _camera_started:
            _picam2_instance.stop()
            _camera_started = False
