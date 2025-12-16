import cv2
from config import frame_size

class CameraStream:
    def start(self) -> None:
        self._cap = cv2.VideoCapture(0)
        frame_size.append(int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        frame_size.append(int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    def capture_frame(self):
        _ , frame = self._cap.read()
        return cv2.flip(frame, 1)
    
    def stop(self):
        self._cap.release()