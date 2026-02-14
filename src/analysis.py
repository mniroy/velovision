import cv2
import face_recognition
import numpy as np
import os
import google.generativeai as genai
from PIL import Image
import threading
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceManager:
    def __init__(self, faces_dir="/data/faces"):
        self.faces_dir = faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_faces()

    def load_faces(self):
        logger.info("Loading known faces...")
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir, exist_ok=True)
            return

        for filename in os.listdir(self.faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(self.faces_dir, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        # Use filename without extension as name
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                except Exception as e:
                    logger.error(f"Error loading face {filename}: {e}")
        logger.info(f"Loaded {len(self.known_face_names)} faces.")

    def add_face(self, name, image_bytes):
        try:
            # Save image to disk
            filename = f"{name}.jpg"
            filepath = os.path.join(self.faces_dir, filename)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            
            # Load and encode
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(name)
                logger.info(f"Added face: {name}")
                return True, "Face added successfully."
            else:
                os.remove(filepath) # Cleanup
                return False, "No face detected in image."
        except Exception as e:
            logger.error(f"Error adding face: {e}")
            return False, str(e)

    def remove_face(self, name):
        try:
            if name in self.known_face_names:
                index = self.known_face_names.index(name)
                self.known_face_names.pop(index)
                self.known_face_encodings.pop(index)
                
                # Try to remove file
                for ext in ['.jpg', '.jpeg', '.png']:
                    filepath = os.path.join(self.faces_dir, f"{name}{ext}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        break
                
                logger.info(f"Removed face: {name}")
                return True, "Face removed."
            return False, "Face not found."
        except Exception as e:
            logger.error(f"Error removing face: {e}")
            return False, str(e)

    def identify_faces(self, frame_rgb):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)
        
        # Find all faces and encodings
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if self.known_face_encodings:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)
        
        return face_names, face_locations

class AIAnalyzer:
    def __init__(self, api_key=None, model_name="gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None
            logger.warning("AI API Key not set. Analysis will be disabled.")

    def analyze_image(self, image_bytes, prompt, faces_detected=[]):
        if not self.model:
            return "AI Analysis Disabled (No API Key)"

        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Construct full prompt
            context = f"Identified people in scene: {', '.join(faces_detected) if faces_detected else 'None'}."
            full_prompt = f"{prompt}\nContext: {context}"
            
            response = self.model.generate_content([full_prompt, image])
            return response.text
        except Exception as e:
            logger.error(f"AI Analysis failed: {e}")
            return f"Error: {str(e)}"

    def analyze_multi_images(self, images_data, global_prompt):
        """
        images_data list of dict: {"image_bytes": b'', "camera_name": "..."}
        """
        if not self.model:
            return "AI Analysis Disabled"
        
        try:
            inputs = [global_prompt]
            for item in images_data:
                img = Image.open(io.BytesIO(item["image_bytes"]))
                inputs.append(f"Camera: {item['camera_name']}")
                inputs.append(img)
            
            response = self.model.generate_content(inputs)
            return response.text
        except Exception as e:
            logger.error(f"Multi-AI Analysis failed: {e}")
            return f"Error: {str(e)}"

# Global Instances (to be initialized by main app)
face_manager = None
ai_analyzer = None

def list_models(api_key=None):
    if not api_key:
        from src.config import config
        api_key = config.get("ai", {}).get("api_key") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return []
    
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name.replace('models/', ''))
        return models
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []

def init_analysis(api_key_env="GEMINI_API_KEY"):
    global face_manager, ai_analyzer
    face_manager = FaceManager()
    
    from src.config import config
    ai_config = config.get("ai", {})
    api_key = ai_config.get("api_key") or os.getenv(api_key_env)
    model_name = ai_config.get("model", "gemini-1.5-flash")
    
    logger.info(f"Initializing AI Analyzer with model: {model_name}")
    ai_analyzer = AIAnalyzer(api_key=api_key, model_name=model_name)
