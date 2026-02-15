import cv2
import time
import threading
import logging
import asyncio

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, source=0, name="Camera"):
        self.source = source
        self.name = name
        self.last_encoded_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.error_count = 0
        self.max_errors = 50  # Stop retrying after this many consecutive failures
        
        try:
            # Handle string indices for USB cameras
            if isinstance(source, str) and source.isdigit():
                source = int(source)
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
        if self.video and self.video.isOpened():
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
                    # Pre-encode frame to JPEG in the background thread
                    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        with self.lock:
                            self.last_encoded_frame = jpeg.tobytes()
                else:
                    self.error_count += 1
                    time.sleep(1.0)
                    # Try to reset capture if failing
                    if self.error_count % 5 == 0:
                        self.video.release()
                        self.video = cv2.VideoCapture(self.source)
                    
            except Exception as e:
                logger.error(f"Camera {self.name} update error: {e}")
                self.error_count += 1
                time.sleep(2.0)
                
            time.sleep(0.01) # Poll reasonably fast

    def get_frame(self):
        with self.lock:
            return self.last_encoded_frame

async def generate_frames(camera, fps=20):
    empty_count = 0
    max_empty = 600  # ~1 min of no frames
    delay = 1.0 / fps
    while True:
        frame = camera.get_frame()
        if frame is not None:
            empty_count = 0
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            await asyncio.sleep(delay)
        else:
            empty_count += 1
            if empty_count > max_empty:
                logger.warning(f"Stopping stream for {camera.name} - no frames")
                break
            await asyncio.sleep(0.1)

# Global Manager
class CameraManager:
    def __init__(self):
        self.cameras = {}

    def get_camera(self, camera_id):
        return self.cameras.get(camera_id)

    def add_camera(self, camera_id, source, name=None):
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
