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
                
                if filename.startswith("._") or filename.startswith("."):
                    try:
                        os.remove(filepath)
                        logger.info(f"Auto-deleted hidden/system file: {filename}")
                    except: pass
                    continue
                
                # Pre-check file size
                if os.path.exists(filepath) and os.path.getsize(filepath) == 0:
                    try:
                        os.remove(filepath)
                        logger.warning(f"Auto-deleted empty file: {filename}")
                    except: pass
                    continue

                try:
                    # 1. Verify with PIL to ensure image integrity
                    try:
                        with Image.open(filepath) as pil_img:
                            pil_img.verify()
                    except Exception as pil_e:
                        logger.error(f"Corrupt image file detected (PIL verify failed) {filename}: {pil_e}")
                        try:
                            os.remove(filepath)
                            logger.info(f"Auto-deleted corrupt file: {filename}")
                        except: pass
                        continue

                    # 2. Add to known faces list (Metadata only, no heavy dlib loading)
                    # We rely on Gemini for recognition, so we don't need local embeddings on startup.
                    name = os.path.splitext(filename)[0]
                    if name not in self.known_face_names:
                        self.known_face_names.append(name)
                        # self.known_face_encodings.append(...) # Skipped for stability & speed
                    
                except Exception as e:
                    logger.error(f"Error registering face {filename}: {e}")
        logger.info(f"Loaded {len(self.known_face_names)} faces.")

    def add_face(self, name, image_bytes):
        try:
            # Save image to disk
            filename = f"{name}.jpg"
            filepath = os.path.join(self.faces_dir, filename)
            
            # Write new file
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            
            # Load and encode
            import face_recognition
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                new_encoding = encodings[0]
                
                if name in self.known_face_names:
                    # Update existing face (remove old encoding, add new)
                    idx = self.known_face_names.index(name)
                    self.known_face_encodings[idx] = new_encoding
                    logger.info(f"Updated face encoding: {name}")
                else:
                    # Add new face
                    self.known_face_encodings.append(new_encoding)
                    self.known_face_names.append(name)
                    logger.info(f"Added new face: {name}")
                    
                return True, "Face added successfully."
            else:
                # No face encoding found, but the image file is still valid.
                # Keep the file on disk and register the name so Gemini can still
                # use it as a reference image for recognition.
                if name not in self.known_face_names:
                    self.known_face_names.append(name)
                    logger.info(f"Added face without encoding (no face detected by face_recognition): {name}")
                
                return True, "Face image saved (no local encoding, AI will handle recognition)."
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
            inputs.append("\nIMPORTANT: Analyze the NEW SCENE and return a list of ALL people detected.")
            inputs.append("For each person, provide:")
            inputs.append("1. 'name': Their name if recognized from reference images, otherwise 'Unknown'.")
            inputs.append("2. 'box_2d': Normalized bounding box [ymin, xmin, ymax, xmax] (0-1000).")
            inputs.append("3. 'status': 'Known' or 'Unknown'.")
            inputs.append("\nFormat your response as a valid JSON block at the beginning: ```json [{\"name\": \"...\", \"box_2d\": [...], \"status\": \"...\"}] ``` followed by your descriptive analysis.")
            inputs.append(main_image)
            
            response = self.model.generate_content(inputs)
            text = response.text
            
            # Simple extraction of "person detected" flag
            detected = "PERSON_DETECTED: YES" in text
            
            # Extract JSON list of people
            detections = []
            try:
                import json
                # Find JSON block
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                    detections = json.loads(json_str)
                elif "[" in text and "]" in text and "name" in text:
                    # Fallback find first [ and last ]
                    start = text.find("[")
                    end = text.rfind("]") + 1
                    detections = json.loads(text[start:end])
            except Exception as je:
                logger.warning(f"Could not parse detections JSON: {je}")

            recognized_names = [d['name'] for d in detections if d.get('status') == 'Known']
            unknown_detections = [d for d in detections if d.get('status') == 'Unknown' or d.get('name') == 'Unknown']
            is_new_person = len(unknown_detections) > 0

            # CLEANUP: Remove JSON block from the human description
            cleaned_text = text
            if "```json" in cleaned_text:
                parts = cleaned_text.split("```")
                # If it starts with JSON, parts[0] is empty, parts[1] is json, parts[2] is text
                if len(parts) >= 3:
                     # Join everything AFTER the JSON block
                     cleaned_text = "".join(parts[2:]).strip()
            elif "[" in text and "]" in text and "name" in text:
                # Fallback for plain formatting
                end = text.rfind("]") + 1
                cleaned_text = text[end:].strip()

            return cleaned_text, is_new_person, recognized_names, len(unknown_detections), detections
        except Exception as e:
            logger.error(f"AI Analysis failed: {e}")
            return f"Error: {str(e)}", False, [], 0, []

    def analyze_multi_images(self, images_data, global_prompt, known_faces=[]):
        """
        images_data list of dict: {"image_bytes": b'', "camera_name": "...", "camera_id": "..."}
        known_faces: list of dict {"name": "...", "image_path": "..."}
        Returns: (summary_text, primary_camera_id, detections_by_camera)
          - primary_camera_id: camera_id with most activity (or None)
          - detections_by_camera: dict {camera_id: [{"name": ..., "status": ...}]}
        """
        if not self.model:
            return "AI Analysis Disabled", None, {}
        
        try:
            inputs = []

            # Add known face references for person recognition
            if known_faces:
                inputs.append("Here are reference images of people I know. Use these to identify anyone in the camera feeds:")
                for face in known_faces:
                    try:
                        if os.path.exists(face['image_path']):
                            face_img = Image.open(face['image_path'])
                            inputs.append(f"This is {face['name']}:")
                            inputs.append(face_img)
                    except Exception as e:
                        logger.warning(f"Could not load reference image for {face['name']}: {e}")

            inputs.append(f"PATROL TASK:\n{global_prompt}")
            
            # Build camera name to ID mapping
            cam_id_map = {}
            for item in images_data:
                img = Image.open(io.BytesIO(item["image_bytes"]))
                cam_label = item['camera_name']
                cam_id_map[cam_label] = item.get('camera_id', cam_label)
                inputs.append(f"Camera: {cam_label}")
                inputs.append(img)
            
            inputs.append("\nIMPORTANT INSTRUCTIONS:")
            inputs.append("1. For each camera, identify ALL people visible. If they match a reference image, state their name. Otherwise mark as 'Unknown'.")
            inputs.append("2. Determine which camera shows the PRIMARY ACTIVITY (most people, most movement, or most noteworthy scene).")
            inputs.append("3. Start your response with a JSON block in this exact format:")
            inputs.append("```json")
            inputs.append("{")
            inputs.append('  "primary_camera": "Camera Name with main activity",')
            inputs.append('  "cameras": {')
            inputs.append('    "Camera Name": [{"name": "PersonName or Unknown", "status": "Known or Unknown"}],')
            inputs.append('    "Camera Name 2": []')
            inputs.append("  }")
            inputs.append("}")
            inputs.append("```")
            inputs.append("4. After the JSON block, provide your descriptive patrol summary. Mention recognized people by name.")
            
            response = self.model.generate_content(inputs)
            text = response.text

            # Parse structured response
            primary_camera_id = None
            detections_by_camera = {}

            try:
                import json
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                    parsed = json.loads(json_str)
                    
                    # Resolve primary camera name to ID
                    primary_name = parsed.get("primary_camera", "")
                    primary_camera_id = cam_id_map.get(primary_name)
                    # Fallback: fuzzy match if exact name doesn't match
                    if not primary_camera_id:
                        for cam_name, cam_id in cam_id_map.items():
                            if cam_name.lower() in primary_name.lower() or primary_name.lower() in cam_name.lower():
                                primary_camera_id = cam_id
                                break

                    # Resolve per-camera detections to use camera IDs
                    cameras_data = parsed.get("cameras", {})
                    for cam_name, people_list in cameras_data.items():
                        resolved_id = cam_id_map.get(cam_name)
                        if not resolved_id:
                            for cn, ci in cam_id_map.items():
                                if cn.lower() in cam_name.lower() or cam_name.lower() in cn.lower():
                                    resolved_id = ci
                                    break
                        if resolved_id:
                            detections_by_camera[resolved_id] = people_list
            except Exception as je:
                logger.warning(f"Could not parse patrol detections JSON: {je}")

            # Clean JSON block from summary text
            cleaned_text = text
            if "```json" in cleaned_text:
                parts = cleaned_text.split("```")
                if len(parts) >= 3:
                    cleaned_text = "".join(parts[2:]).strip()

            return cleaned_text, primary_camera_id, detections_by_camera
        except Exception as e:
            logger.error(f"Multi-AI Analysis failed: {e}")
            return f"Error: {str(e)}", None, {}

    def find_persons(self, target_faces, images_data, custom_prompt=""):
        """
        Search for specific people across multiple camera feeds.
        target_faces: list of dict {"name": "...", "image_path": "..."}
        images_data: list of dict {"image_bytes": b'', "camera_name": "...", "camera_id": "..."}
        custom_prompt: additional instructions for the AI
        Returns: (summary_text, results_by_camera)
          results_by_camera: dict {camera_id: {"camera_name": ..., "found": [{"name": ..., "activity": ..., "confidence": ...}], "not_found": [...]}}
        """
        if not self.model:
            return "AI Analysis Disabled", {}

        if not target_faces:
            return "No target persons specified.", {}

        try:
            inputs = []
            target_names = [f['name'] for f in target_faces]

            # Add target person reference images
            inputs.append(f"I am looking for these specific people. Find them in the camera feeds below:")
            for face in target_faces:
                try:
                    if os.path.exists(face['image_path']):
                        face_img = Image.open(face['image_path'])
                        inputs.append(f"TARGET: {face['name']}")
                        inputs.append(face_img)
                except Exception as e:
                    logger.warning(f"Could not load target image for {face['name']}: {e}")

            # Add custom prompt if provided
            if custom_prompt:
                inputs.append(f"\nADDITIONAL INSTRUCTIONS: {custom_prompt}")

            # Add camera feeds
            cam_id_map = {}
            for item in images_data:
                img = Image.open(io.BytesIO(item["image_bytes"]))
                cam_label = item['camera_name']
                cam_id_map[cam_label] = item.get('camera_id', cam_label)
                inputs.append(f"Camera: {cam_label}")
                inputs.append(img)

            inputs.append(f"\nTASK: Search for the target people ({', '.join(target_names)}) in ALL camera feeds above.")
            inputs.append("For each camera, report whether each target person is found or not.")
            inputs.append("If found, describe what they are doing and their location within the scene.")
            inputs.append("Start your response with a JSON block in this exact format:")
            inputs.append("```json")
            inputs.append("{")
            inputs.append('  "cameras": {')
            inputs.append('    "Camera Name": {')
            inputs.append('      "found": [{"name": "PersonName", "activity": "description of what they are doing", "confidence": "high/medium/low"}],')
            inputs.append('      "not_found": ["PersonName2"]')
            inputs.append("    }")
            inputs.append("  }")
            inputs.append("}")
            inputs.append("```")
            inputs.append("After the JSON block, provide a natural language summary of where each person was found and what they are doing.")

            response = self.model.generate_content(inputs)
            text = response.text

            # Parse structured response
            results_by_camera = {}
            try:
                import json
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                    parsed = json.loads(json_str)

                    cameras_data = parsed.get("cameras", {})
                    for cam_name, cam_results in cameras_data.items():
                        resolved_id = cam_id_map.get(cam_name)
                        if not resolved_id:
                            for cn, ci in cam_id_map.items():
                                if cn.lower() in cam_name.lower() or cam_name.lower() in cn.lower():
                                    resolved_id = ci
                                    break
                        if resolved_id:
                            results_by_camera[resolved_id] = {
                                "camera_name": cam_name,
                                "found": cam_results.get("found", []),
                                "not_found": cam_results.get("not_found", [])
                            }
            except Exception as je:
                logger.warning(f"Could not parse person finder JSON: {je}")

            # Clean JSON block from summary text
            cleaned_text = text
            if "```json" in cleaned_text:
                parts = cleaned_text.split("```")
                if len(parts) >= 3:
                    cleaned_text = "".join(parts[2:]).strip()

            return cleaned_text, results_by_camera
        except Exception as e:
            logger.error(f"Person Finder AI failed: {e}")
            return f"Error: {str(e)}", {}

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
