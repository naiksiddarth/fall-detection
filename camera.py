# camera.py
from picamera2 import Picamera2

class CameraStream:
    def __init__(self, width=640, height=480):
        print("Starting camera feed...")
        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam2.configure(self.config)
        
    def start(self):
        """Starts the camera stream."""
        self.picam2.start()

    def capture_frame(self):
        """Captures a single frame from the camera."""
        # capture_array returns a NumPy array
        return self.picam2.capture_array()

    def stop(self):
        """Stops the camera stream."""
        self.picam2.stop()
        print("Camera feed stopped.")