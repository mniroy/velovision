import cv2
import time
import threading
import logging

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, source=0, name="Camera"):
        self.source = source
        self.name = name
        self.last_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.error_count = 0
        self.max_errors = 50  # Stop retrying after this many consecutive failures
        
        try:
            self.video = cv2.VideoCapture(self.source)
            if not self.video.isOpened():
                logger.error(f"Could not open video source {source}")
        except Exception as e:
            logger.error(f"Failed to create VideoCapture for {source}: {e}")
            self.video = None
        
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
            try:
                if self.video is None or not self.video.isOpened():
                    self.error_count += 1
                    if self.error_count > self.max_errors:
                        logger.error(f"Camera {self.name}: Max errors reached. Stopping retry loop.")
                        self.running = False
                        break
                    time.sleep(5.0)  # Wait before retry
                    try:
                        self.video = cv2.VideoCapture(self.source)
                    except Exception:
                        pass
                    continue
                
                success, frame = self.video.read()
                if success:
                    self.error_count = 0  # Reset on success
                    with self.lock:
                        self.last_frame = frame
                else:
                    self.error_count += 1
                    time.sleep(2.0)
                    
            except Exception as e:
                logger.error(f"Camera {self.name} update error: {e}")
                self.error_count += 1
                time.sleep(2.0)
                
            time.sleep(0.03)

    def get_frame(self):
        with self.lock:
            if self.last_frame is not None and self.last_frame.size > 0:
                try:
                    ret, jpeg = cv2.imencode('.jpg', self.last_frame)
                    if ret:
                        return jpeg.tobytes()
                except Exception as e:
                    logger.error(f"Frame encoding error: {e}")
            return None

def generate_frames(camera):
    empty_count = 0
    max_empty = 300  # ~30 seconds of no frames
    while True:
        frame = camera.get_frame()
        if frame is not None:
            empty_count = 0
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            empty_count += 1
            if empty_count > max_empty:
                break  # Stop streaming if camera is dead
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
