import cv2

class CameraStream:
    def start(self) -> None:
        self._cap = cv2.VideoCapture(0)
    
    def capture_frame(self):
        _ , frame = self._cap.read()
        return cv2.flip(frame, 1)
    
    def stop(self):
        self._cap.release()