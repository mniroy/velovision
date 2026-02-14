import cv2
# import face_recognition  # Deferred to lazy loading for low-RAM stability
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
                    import face_recognition
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
            import face_recognition
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

    def get_known_faces(self):
        """Returns metadata for all labeled faces."""
        faces = []
        for name in self.known_face_names:
            for ext in ['.jpg', '.jpeg', '.png']:
                filepath = os.path.join(self.faces_dir, f"{name}{ext}")
                if os.path.exists(filepath):
                    faces.append({"name": name, "image_path": filepath})
                    break
        return faces

    def identify_faces(self, frame_rgb):
        """Bypassed: Using Cloud AI (Gemini) for all person detection to save resources."""
        return [], []

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

    def analyze_image(self, image_bytes, prompt, known_faces=[]):
        """
        known_faces: list of dict {"name": "...", "image_path": "..."}
        """
        if not self.model:
            return "AI Analysis Disabled (No API Key)", []

        try:
            # Main image
            main_image = Image.open(io.BytesIO(image_bytes))
            
            inputs = []
            
            # Add context about known people with their images
            if known_faces:
                inputs.append("Here are reference images of people I know. Use these to identify them in the new scene:")
                for face in known_faces:
                    try:
                        if os.path.exists(face['image_path']):
                            face_img = Image.open(face['image_path'])
                            inputs.append(f"This is {face['name']}:")
                            inputs.append(face_img)
                    except Exception as e:
                        logger.warning(f"Could not load reference image for {face['name']}: {e}")

            inputs.append(f"NEW SCENE TO ANALYZE:\n{prompt}")
            inputs.append("\nIMPORTANT:")
            inputs.append("1. If you see someone you don't recognize among the known people, start your response with 'PERSON_DETECTED: YES'. Otherwise start with 'PERSON_DETECTED: NO'.")
            inputs.append("2. If you identify any known people from the reference images, list their names exactly as provided in the format 'RECOGNIZED_PEOPLE: Name1, Name2'.")
            inputs.append("3. Count how many people are in the image that you CANNOT identify from the reference images. List this count as 'UNKNOWN_COUNT: X'.")
            inputs.append("4. Provide the descriptive analysis after these tags.")
            inputs.append(main_image)
            
            response = self.model.generate_content(inputs)
            text = response.text
            
            # Simple extraction of "person detected" flag
            detected = "PERSON_DETECTED: YES" in text
            
            # Extract recognized people
            recognized_names = []
            unknown_count = 0
            
            lines = text.split('\n')
            for line in lines:
                if "RECOGNIZED_PEOPLE:" in line:
                    try:
                        names_str = line.split("RECOGNIZED_PEOPLE:")[1].strip()
                        recognized_names = [n.strip() for n in names_str.split(",") if n.strip()]
                    except: pass
                if "UNKNOWN_COUNT:" in line:
                    try:
                        count_str = line.split("UNKNOWN_COUNT:")[1].strip()
                        # Extract digits only
                        import re
                        match = re.search(r'\d+', count_str)
                        if match:
                            unknown_count = int(match.group())
                    except: pass
            
            # Final detected flag based on Gemini's count if available, otherwise fallback to its own YES/NO
            is_new_person = unknown_count > 0 or ("PERSON_DETECTED: YES" in text)

            return text, is_new_person, recognized_names, unknown_count
        except Exception as e:
            logger.error(f"AI Analysis failed: {e}")
            return f"Error: {str(e)}", False, [], 0

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
