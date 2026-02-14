import cv2
import time
import threading
import logging

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, source=0, name="Camera"):
        self.source = source
        self.name = name
        self.video = cv2.VideoCapture(self.source)
        if not self.video.isOpened():
            logger.error(f"Could not open video source {source}")
            # Fallback to 0 if string source fails? No.
        
        self.last_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __del__(self):
        self.stop()

    def stop(self):
        self.running = False
        if self.video.isOpened():
            self.video.release()

    def _update(self):
        while self.running:
            success, frame = self.video.read()
            if success:
                with self.lock:
                    self.last_frame = frame
            else:
                # Retry logic for network streams?
                time.sleep(1.0)
                if not self.video.isOpened():
                     self.video.open(self.source)
                
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.last_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', self.last_frame)
                return jpeg.tobytes()
            return None

def generate_frames(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            time.sleep(0.1)

# Global Manager
class CameraManager:
    def __init__(self):
        self.cameras = {}

    def get_camera(self, camera_id):
        return self.cameras.get(camera_id)

    def add_camera(self, camera_id, source, name=None):
        # Stop existing if any (re-add/update)
        if camera_id in self.cameras:
            try:
                self.cameras[camera_id].stop()
            except:
                pass
        
        cam = Camera(source=source, name=name or camera_id)
        self.cameras[camera_id] = cam
        return cam

    def remove_camera(self, camera_id):
        if camera_id in self.cameras:
            try:
                self.cameras[camera_id].stop()
            except:
                pass
            del self.cameras[camera_id]
            return True
        return False

camera_manager = CameraManager()
