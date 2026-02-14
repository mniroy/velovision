from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import APIRouter, BackgroundTasks
import logging
from src.streaming import camera_manager
from src import analysis, whatsapp
from src.config import config
import cv2
import numpy as np
import time
import os
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()
scheduler = BackgroundScheduler()

# Initialize Analysis components
def start_scheduler():
    analysis.init_analysis()
    whatsapp.init_whatsapp()
    if not scheduler.running:
        scheduler.start()
    sync_schedules()
    logger.info("Scheduler started.")

@router.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()
    logger.info("Scheduler shut down.")

def sync_schedules():
    """Sync all camera schedules from config to the scheduler."""
    for cam_id, cam_config in config.get("cameras", {}).items():
        if cam_config.get("enabled", True) and cam_config.get("schedule_enabled"):
            hrs = int(cam_config.get("schedule_interval_hrs", 1))
            mins = int(cam_config.get("schedule_interval_mins", 0))
            
            # Skip if interval is 0
            if hrs == 0 and mins == 0:
                continue
                
            job_id = f"analysis_{cam_id}"
            if scheduler.get_job(job_id):
                scheduler.remove_job(job_id)
            
            scheduler.add_job(
                perform_analysis,
                'interval',
                hours=hrs,
                minutes=mins,
                args=[cam_id],
                id=job_id
            )
            logger.info(f"Scheduled analysis for {cam_id} every {hrs}h {mins}m")
        else:
            # Remove job if disabled
            job_id = f"analysis_{cam_id}"
            if scheduler.get_job(job_id):
                scheduler.remove_job(job_id)

    # Global Patrol Schedule
    patrol_config = config.get("patrol", {})
    job_id = "patrol_global"
    if patrol_config.get("schedule_enabled"):
        hrs = int(patrol_config.get("schedule_interval_hrs", 6))
        if hrs > 0:
            if scheduler.get_job(job_id):
                scheduler.remove_job(job_id)
            scheduler.add_job(
                perform_home_patrol,
                'interval',
                hours=hrs,
                id=job_id
            )
            logger.info(f"Scheduled GLOBAL PATROL every {hrs}h")
    else:
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)

def perform_analysis(camera_id="default"):
    logger.info(f"Starting analysis for camera: {camera_id}")
    
    # 1. Capture Frame
    try:
        camera = camera_manager.get_camera(camera_id)
        if not camera:
             logger.error(f"Camera {camera_id} not found.")
             return {"status": "error", "message": "Camera not found"}
        
        frame_bytes = camera.get_frame()

        if not frame_bytes:
            logger.error("Failed to capture frame.")
            return {"status": "error", "message": "Failed to capture frame"}

        # Decode for processing
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Face Recognition
        face_names, locations = analysis.face_manager.identify_faces(rgb_frame)
        logger.info(f"Detected faces: {face_names}")

        # 3. AI Analysis
        # Fetch camera config
        cam_config = config.get("cameras", {}).get(camera_id, {})
        default_prompt = "Describe the current scene. If there are people, mention them by name if known or describe their appearance."
        prompt = cam_config.get("analysis_prompt", default_prompt)
        
        # Incorporate message instruction if available
        msg_instruction = cam_config.get("message_instruction")
        if msg_instruction:
            prompt = f"{prompt}\nInstruction for delivery: {msg_instruction}"

        # Get known faces for AI context
        known_faces = analysis.face_manager.get_known_faces()
        
        analysis_result, person_detected, recognized_names, unknown_count, detections = analysis.ai_analyzer.analyze_image(frame_bytes, prompt, known_faces=known_faces)
        logger.info(f"AI Analysis ({camera_id}): {len(detections)} people detected. Recognized: {recognized_names}, Unknown: {unknown_count}")

        # 4. Save Event to DB
        timestamp = datetime.now()
        filename = f"{camera_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join("/data/events", filename)
        
        # Save main image
        with open(filepath, "wb") as f:
            f.write(frame_bytes)

        # Save to DB
        from src.database import SessionLocal, Event, UnlabeledPerson, Face
        db = SessionLocal()
        try:
            event = Event(
                timestamp=timestamp,
                camera_id=camera_id,
                image_path=filepath,
                analysis_text=analysis_result,
                faces_detected=",".join(recognized_names),
                prompt_used=prompt
            )
            db.add(event)
            db.flush() # Get event ID

            # Update sighting stats for recognized people
            for name in recognized_names:
                face = db.query(Face).filter(Face.name == name).first()
                if face:
                    face.last_seen = timestamp
                    face.sighting_count = (face.sighting_count or 0) + 1
            
            # Process ALL unknown detections individually
            from PIL import Image
            import io
            main_img = Image.open(io.BytesIO(frame_bytes))
            width, height = main_img.size

            unknown_list = [d for d in detections if d.get('status') == 'Unknown' or d.get('name') == 'Unknown']
            
            for i, det in enumerate(unknown_list):
                try:
                    box = det.get('box_2d') # [ymin, xmin, ymax, xmax] 0-1000
                    if box and len(box) == 4:
                        # Extract and Crop
                        left = box[1] * width / 1000
                        top = box[0] * height / 1000
                        right = box[3] * width / 1000
                        bottom = box[2] * height / 1000
                        
                        # Add some padding around the face (20%)
                        padding_w = (right - left) * 0.2
                        padding_h = (bottom - top) * 0.2
                        left = max(0, left - padding_w)
                        top = max(0, top - padding_h)
                        right = min(width, right + padding_w)
                        bottom = min(height, bottom + padding_h)

                        face_img = main_img.crop((left, top, right, bottom))
                        
                        # Save cropped face
                        face_filename = f"crop_{event.id}_{i}.jpg"
                        face_path = os.path.join("/data/faces", face_filename)
                        face_img.save(face_path, "JPEG")
                        
                        unlabeled = UnlabeledPerson(
                            image_path=face_path,
                            timestamp=timestamp,
                            camera_id=camera_id,
                            event_id=event.id
                        )
                        db.add(unlabeled)
                        logger.info(f"Added isolated face discovery {i} to queue.")
                    else:
                        # Fallback to full image if no box
                        unlabeled = UnlabeledPerson(
                            image_path=filepath,
                            timestamp=timestamp,
                            camera_id=camera_id,
                            event_id=event.id
                        )
                        db.add(unlabeled)
                except Exception as ce:
                    logger.error(f"Failed to isolate face {i}: {ce}")

            db.commit()
            logger.info(f"Event saved to DB: ID {event.id}")
        except Exception as e:
            logger.error(f"DB Error: {e}")
        finally:
            db.close()
        
        # 5. Send WhatsApp Notification
        whatsapp_status = {"enabled": False, "sent": False, "recipients": 0}
        
        if whatsapp.client:
            whatsapp_status["enabled"] = True
            # ONLY use explicitly assigned recipients
            recipients = cam_config.get("recipients")
            
            if recipients:
                name = cam_config.get("name", camera_id)
                caption = f"🚨 *Velo Vision Alert: {name}*\n\n{analysis_result}"
                results = whatsapp.client.send_alert(recipients, frame_bytes, caption)
                
                # Log notifications to DB
                from src.database import Notification
                db = SessionLocal()
                try:
                    for res in results:
                        notif = Notification(
                            event_id=event.id, # Link back to the analysis event
                            recipient_name=res['name'],
                            recipient_value=res['value'],
                            status="success" if res['success'] else "failed",
                            timestamp=timestamp
                        )
                        db.add(notif)
                    db.commit()
                    whatsapp_status["sent"] = True
                    whatsapp_status["recipients"] = len(results)
                except Exception as e:
                    logger.error(f"Error logging notifications: {e}")
                finally:
                    db.close()

        return {
            "status": "success",
            "timestamp": timestamp.isoformat(),
            "camera": camera_id,
            "faces": face_names,
            "analysis": analysis_result,
            "whatsapp": whatsapp_status
        }

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/trigger/{camera_id}")
def trigger_analysis_manual(camera_id: str):
    """
    Manually trigger analysis and wait for result.
    """
    result = perform_analysis(camera_id)
    return result

@router.post("/patrol/summarize")
def patrol_summarize():
    """
    Home Patrol: Capture snapshots from all cameras and summarize.
    """
    return perform_home_patrol()

def perform_home_patrol():
    logger.info("Executing HOME PATROL...")
    images_data = []
    
    for cam_id, cam_config in config.get("cameras", {}).items():
        if not cam_config.get("enabled", True):
            continue
            
        camera = camera_manager.get_camera(cam_id)
        if camera:
            frame_bytes = camera.get_frame()
            if frame_bytes:
                images_data.append({
                    "image_bytes": frame_bytes,
                    "camera_name": cam_config.get("name", cam_id)
                })
    
    if not images_data:
        return {"status": "error", "message": "No active cameras to capture"}
    
    patrol_config = config.get("patrol", {})
    prompt = patrol_config.get("prompt", "Perform a holistic security patrol of the entire property using these camera snapshots. Summarize the state of the home. If everything is normal, say 'All Clear'. If there are any anomalies or people detected, describe them clearly.")
    
    msg_instruction = patrol_config.get("message_instruction")
    if msg_instruction:
        prompt = f"{prompt}\nInstruction for delivery: {msg_instruction}"

    summary = analysis.ai_analyzer.analyze_multi_images(images_data, prompt)
    logger.info(f"Home Patrol Summary: {summary}")
    
    # Send to global recipients
    whatsapp_status = {"sent": False, "recipients": 0}
    
    if whatsapp.client:
        # ONLY use explicitly assigned recipients
        recipients = patrol_config.get("recipients")
            
        if recipients:
            caption = f"🛡️ *Home Patrol Summary*\n\n{summary}"
            # Use the first camera image as representative or just send text?
            # Let's send without image for multi-summary for now, or use first one.
            results = whatsapp.client.send_alert(recipients, images_data[0]["image_bytes"], caption)
            whatsapp_status["sent"] = True
            whatsapp_status["recipients"] = len(results)
            
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "whatsapp": whatsapp_status
    }

@router.post("/schedule/{camera_id}")
def schedule_analysis(camera_id: str, interval_minutes: int):
    """
    Schedule periodic analysis.
    """
    job_id = f"analysis_{camera_id}"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)
    
    scheduler.add_job(
        perform_analysis, 
        'interval', 
        minutes=interval_minutes, 
        args=[camera_id], 
        id=job_id
    )
    return {"status": "Scheduled", "job_id": job_id, "interval_minutes": interval_minutes}
