import cv2
import time
import threading
import logging
import asyncio
import os

# Set environment variables for better RTSP performance globally
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, source=0, name="Camera"):
        # Handle string indices for USB cameras
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        self.source = source
        self.name = name
        self.last_raw_frame = None
        self.last_encoded_frame = None
        self.last_frame_id = 0
        self.last_encoded_id = -1
        self.lock = threading.Lock()
        self.running = True
        self.error_count = 0
        self.max_errors = 50 
        self.video = None
        self.last_update_time = 0
        self.last_request_time = time.time()
        
        # Start background thread
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Camera {self.name}: Background thread started.")

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
                    time.sleep(1.0)
                    backend = cv2.CAP_FFMPEG if (isinstance(self.source, str) and "://" in self.source) else None
                    self.video = cv2.VideoCapture(self.source, backend)
                    if self.video.isOpened():
                        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        logger.info(f"Camera {self.name}: Connected.")
                    continue
                
                # GRAB the frame (lightweight, doesn't decode yet)
                if not self.video.grab():
                    self.error_count += 1
                    time.sleep(0.1)
                    continue

                # Only RETRIEVE (decode) if someone has requested a frame in the last 10 seconds
                if (time.time() - self.last_request_time) < 10.0:
                    success, frame = self.video.retrieve()
                    if success:
                        with self.lock:
                            self.last_raw_frame = frame
                            self.last_frame_id += 1
                            self.last_update_time = time.time()
                else:
                    # Idle mode: still grab to keep buffer clean, but don't decode
                    pass
                    
            except Exception as e:
                logger.error(f"Camera {self.name} update error: {e}")
                time.sleep(1.0)
            
            time.sleep(0.01) # Small sleep to prevent CPU spinning

    def get_frame(self):
        self.last_request_time = time.time()
        with self.lock:
            if self.last_raw_frame is None:
                return self.last_encoded_frame

            # If we've already encoded this specific frame, return the cache
            if self.last_encoded_id == self.last_frame_id:
                return self.last_encoded_frame

            # Otherwise, encode to JPEG now (on demand)
            ret, jpeg = cv2.imencode('.jpg', self.last_raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                self.last_encoded_frame = jpeg.tobytes()
                self.last_encoded_id = self.last_frame_id
                return self.last_encoded_frame
        return None


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
