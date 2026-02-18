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
        self.last_encoded_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.error_count = 0
        self.max_errors = 50  # Stop retrying after this many consecutive failures
        self.video = None
        self.last_update_time = 0
        
        # Start background thread immediately; it will handle the first connection
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Camera {self.name}: Background thread started for source {source}")

    def __del__(self):
        self.stop()

    def stop(self):
        self.running = False
        if self.video and self.video.isOpened():
            self.video.release()

    def _update(self):
        first_run = True
        while self.running:
            try:
                if self.video is None or not self.video.isOpened():
                    if not first_run:
                        # Wait before retry, but not on the very first attempt
                        sleep_time = min(5.0, 1.0 + (self.error_count * 0.5))
                        time.sleep(sleep_time)
                    
                    first_run = False
                    self.error_count += 1
                    
                    if self.error_count > self.max_errors:
                        logger.error(f"Camera {self.name}: Max errors reached ({self.max_errors}). Stopping retry loop.")
                        self.running = False
                        break
                    
                    try:
                        # Use FFMPEG backend for RTSP sources
                        backend = cv2.CAP_FFMPEG if (isinstance(self.source, str) and "://" in self.source) else None
                        self.video = cv2.VideoCapture(self.source, backend)
                        
                        if self.video.isOpened():
                            # Optimize for low latency and fast connection
                            self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            # Timeouts help prevent hangs with unresponsive cameras
                            self.video.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
                            self.video.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
                            logger.info(f"Camera {self.name}: Connection established.")
                            self.error_count = 0 
                        else:
                            if self.error_count % 10 == 0:
                                logger.warning(f"Camera {self.name}: Still attempting to connect (attempt {self.error_count})...")
                    except Exception as connection_e:
                        logger.debug(f"Camera {self.name} connection exception: {connection_e}")
                    
                    continue
                
                # GRAB the frame immediately to clear the buffer
                # This ensures we are always looking at the newest data available in the socket
                if not self.video.grab():
                    self.error_count += 1
                    if self.error_count % 5 == 0:
                        self.video.release()
                    time.sleep(0.5)
                    continue

                # RETRIEVE and encode
                # We do this as fast as the grab succeeds
                success, frame = self.video.retrieve()
                if success:
                    self.error_count = 0 
                    # JPEG Quality 80 is a good balance for AI
                    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ret:
                        with self.lock:
                            self.last_encoded_frame = jpeg.tobytes()
                            self.last_update_time = time.time()
                else:
                    self.error_count += 1
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Camera {self.name} update error: {e}")
                self.error_count += 1
                time.sleep(1.0)
                
            # No sleep here, or a very tiny one, to keep the buffer drained
            # grab() will naturally block if there's no new frame yet.
            time.sleep(0.001) 

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
