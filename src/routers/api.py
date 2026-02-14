from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Request, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, FileResponse
from src.streaming import camera_manager, generate_frames
from src.config import config, save_config
from src import analysis, triggers, whatsapp
from sqlalchemy.orm import Session
from src.database import get_db, Event, Face, SessionLocal, UnlabeledPerson, engine
import shutil
import logging
import os
import json
import cv2

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize cameras from config on module load
for cam_id, cam_config in config.get("cameras", {}).items():
    if cam_config.get("enabled", True):
        source = cam_config.get("source", 0)
        name = cam_config.get("name", cam_id)
        logger.info(f"Adding camera: {cam_id} (source: {source})")
        camera_manager.add_camera(cam_id, source, name=name)

@router.get("/video_feed")
def video_feed(camera_id: str = "default"):
    cam = camera_manager.get_camera(camera_id)
    if not cam:
        cameras = list(camera_manager.cameras.values())
        if cameras:
            cam = cameras[0]
        else:
            raise HTTPException(status_code=404, detail="Camera not found")
            
    return StreamingResponse(generate_frames(cam), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/snapshot")
def get_snapshot(camera_id: str = "default"):
    cam = camera_manager.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    frame = cam.get_frame()
    if not frame:
        raise HTTPException(status_code=503, detail="Frame capture failed")
        
    from fastapi.responses import Response
    return Response(content=frame, media_type="image/jpeg")

@router.get("/cameras")
def list_cameras():
    return [{"id": c_id, "name": cam.name} for c_id, cam in camera_manager.cameras.items()]

@router.get("/events")
def get_events(limit: int = 20, offset: int = 0, db: Session = Depends(get_db)):
    events = db.query(Event).order_by(Event.timestamp.desc()).limit(limit).offset(offset).all()
    results = []
    for event in events:
        image_url = ""
        if event.image_path:
            filename = os.path.basename(event.image_path)
            image_url = f"/events/{filename}"
            
        results.append({
            "id": event.id,
            "timestamp": event.timestamp,
            "camera_id": event.camera_id,
            "image_url": image_url,
            "analysis_text": event.analysis_text,
            "faces": event.faces_detected.split(",") if event.faces_detected else [],
            "prompt": event.prompt_used
        })
    return results

@router.get("/faces")
def get_faces(db: Session = Depends(get_db)):
    if not analysis.face_manager:
        raise HTTPException(status_code=503, detail="Face Manager not initialized")
    names = analysis.face_manager.known_face_names
    faces = []
    for name in names:
        face_record = db.query(Face).filter(Face.name == name).first()
        category = face_record.category if face_record and face_record.category else "Uncategorized"
        faces.append({"name": name, "category": category})
    return {"names": names, "faces": faces}

@router.get("/faces/categories")
def get_face_categories(db: Session = Depends(get_db)):
    """Return all distinct categories used across faces."""
    from sqlalchemy import func
    categories = db.query(Face.category).filter(Face.category != None).distinct().all()
    cat_list = [c[0] for c in categories if c[0]]
    defaults = ["Family", "Friend", "Courier", "Neighbor", "Staff", "Uncategorized"]
    for d in defaults:
        if d not in cat_list:
            cat_list.append(d)
    return {"categories": sorted(cat_list)}

@router.put("/faces/{name}/category")
def update_face_category(name: str, data: dict, db: Session = Depends(get_db)):
    """Update the category for a face."""
    category = data.get("category", "Uncategorized")
    face = db.query(Face).filter(Face.name == name).first()
    if not face:
        face = Face(name=name, category=category)
        db.add(face)
    else:
        face.category = category
    db.commit()
    return {"status": "success", "name": name, "category": category}

@router.post("/faces")
async def add_face(name: str, category: str = "Uncategorized", file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not analysis.face_manager:
        raise HTTPException(status_code=503, detail="Face Manager not initialized")
    content = await file.read()
    success, message = analysis.face_manager.add_face(name, content)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    # Save category to DB
    face = db.query(Face).filter(Face.name == name).first()
    if not face:
        face = Face(name=name, category=category)
        db.add(face)
    else:
        face.category = category
    db.commit()
    return {"status": "success", "message": message}

@router.delete("/faces/{name}")
def delete_face(name: str):
    if not analysis.face_manager:
        raise HTTPException(status_code=503, detail="Face Manager not initialized")
    success, message = analysis.face_manager.remove_face(name)
    if not success:
        raise HTTPException(status_code=404, detail=message)
    return {"status": "success", "message": message}

# ─── Unlabeled Persons (AI Detection) ────────────────────────────────────────

@router.get("/faces/unlabeled")
def get_unlabeled_persons(db: Session = Depends(get_db)):
    from src.database import UnlabeledPerson
    unlabeled = db.query(UnlabeledPerson).order_by(UnlabeledPerson.timestamp.desc()).all()
    return [{
        "id": p.id,
        "timestamp": p.timestamp.isoformat(),
        "camera_id": p.camera_id,
        "event_id": p.event_id
    } for p in unlabeled]

@router.get("/faces/unlabeled/{id}/image")
def get_unlabeled_image(id: int, db: Session = Depends(get_db)):
    from src.database import UnlabeledPerson
    person = db.query(UnlabeledPerson).filter(UnlabeledPerson.id == id).first()
    if not person or not os.path.exists(person.image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(person.image_path)

@router.post("/faces/unlabeled/{id}/label")
async def label_person(id: int, name: str = Form(...), category: str = Form("Uncategorized"), db: Session = Depends(get_db)):
    from src.database import UnlabeledPerson, Face
    person = db.query(UnlabeledPerson).filter(UnlabeledPerson.id == id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    # Check if face already exists
    existing = db.query(Face).filter(Face.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail="A person with this name already exists")
    
    # Capture the image from the unlabeled entry
    if not os.path.exists(person.image_path):
         raise HTTPException(status_code=404, detail="Reference image missing")
    
    with open(person.image_path, "rb") as f:
        content = f.read()
    
    # 1. Add to FaceManager (disk storage)
    success, message = analysis.face_manager.add_face(name, content)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    # 2. Add to database
    new_face = Face(
        name=name,
        category=category,
        image_path=os.path.join("/data/faces", f"{name}.jpg"),
        last_seen=person.timestamp
    )
    db.add(new_face)
    
    # 3. Cleanup: Remove the unlabeled record (and others if we want to deduplicate? No, just this one)
    db.delete(person)
    db.commit()
    
    return {"status": "success", "message": f"Identified as {name}"}

@router.delete("/faces/unlabeled/{id}")
def delete_unlabeled(id: int, db: Session = Depends(get_db)):
    from src.database import UnlabeledPerson
    person = db.query(UnlabeledPerson).filter(UnlabeledPerson.id == id).first()
    if person:
        db.delete(person)
        db.commit()
    return {"status": "success"}

@router.delete("/events/{event_id}")
def delete_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    if event.image_path and os.path.exists(event.image_path):
        os.remove(event.image_path)
    db.delete(event)
    db.commit()
    return {"status": "deleted", "id": event_id}

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    active_cameras = len(camera_manager.cameras)
    total_events = db.query(Event).count()
    latest_event = db.query(Event).order_by(Event.timestamp.desc()).first()
    latest_event_data = None
    if latest_event:
        image_url = f"/events/{os.path.basename(latest_event.image_path)}" if latest_event.image_path else None
        faces = [f.strip() for f in latest_event.faces_detected.split(',')] if latest_event.faces_detected else []
        latest_event_data = {
            "id": latest_event.id,
            "timestamp": latest_event.timestamp.isoformat(),
            "image_url": image_url,
            "faces": faces,
            "camera_id": latest_event.camera_id,
            "analysis_text": latest_event.analysis_text or ""
        }
    return {
        "active_cameras": active_cameras,
        "total_events": total_events,
        "latest_event": latest_event_data
    }

@router.get("/activity")
def get_activity_log(period: str = "week", category: str = "all", db: Session = Depends(get_db)):
    """Returns event counts grouped by time period. Supports period=week|month|ytd|all and category filtering."""
    from sqlalchemy import func
    from datetime import datetime, timedelta
    
    today = datetime.now().date()
    
    # Build face filter for category
    face_names = None
    if category and category != "all":
        faces_in_cat = db.query(Face.name).filter(Face.category == category).all()
        face_names = [f[0] for f in faces_in_cat]
    
    if period == "week":
        days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
        results = []
        for day in days:
            query = db.query(Event).filter(func.date(Event.timestamp) == day)
            if face_names is not None:
                from sqlalchemy import or_
                query = query.filter(or_(*[Event.faces_detected.contains(n) for n in face_names])) if face_names else query.filter(Event.id < 0)
            results.append({
                "label": day.strftime("%a").upper(),
                "count": query.count(),
                "is_today": day == today
            })
        return results
    
    elif period == "month":
        start = today.replace(day=1)
        import calendar
        _, last_day = calendar.monthrange(today.year, today.month)
        weeks = []
        current = start
        week_num = 1
        while current <= today:
            week_end = min(current + timedelta(days=6), today)
            query = db.query(Event).filter(
                func.date(Event.timestamp) >= current,
                func.date(Event.timestamp) <= week_end
            )
            if face_names is not None:
                from sqlalchemy import or_
                query = query.filter(or_(*[Event.faces_detected.contains(n) for n in face_names])) if face_names else query.filter(Event.id < 0)
            weeks.append({
                "label": f"W{week_num}",
                "count": query.count(),
                "is_today": current <= today <= week_end
            })
            current = week_end + timedelta(days=1)
            week_num += 1
        return weeks
    
    elif period == "ytd":
        start = today.replace(month=1, day=1)
        results = []
        for m in range(1, today.month + 1):
            query = db.query(Event).filter(
                func.strftime('%Y', Event.timestamp) == str(today.year),
                func.strftime('%m', Event.timestamp) == f"{m:02d}"
            )
            if face_names is not None:
                from sqlalchemy import or_
                query = query.filter(or_(*[Event.faces_detected.contains(n) for n in face_names])) if face_names else query.filter(Event.id < 0)
            import calendar
            results.append({
                "label": calendar.month_abbr[m].upper(),
                "count": query.count(),
                "is_today": m == today.month
            })
        return results
    
    else:  # all time - group by month across all years
        from sqlalchemy import func as sqlfunc
        min_date = db.query(sqlfunc.min(Event.timestamp)).scalar()
        if not min_date:
            return []
        start = min_date.date().replace(day=1)
        results = []
        current = start
        while current <= today:
            import calendar
            _, last_day = calendar.monthrange(current.year, current.month)
            month_end = current.replace(day=last_day)
            query = db.query(Event).filter(
                func.date(Event.timestamp) >= current,
                func.date(Event.timestamp) <= month_end
            )
            if face_names is not None:
                from sqlalchemy import or_
                query = query.filter(or_(*[Event.faces_detected.contains(n) for n in face_names])) if face_names else query.filter(Event.id < 0)
            results.append({
                "label": current.strftime("%b %y").upper() if current.year != today.year else current.strftime("%b").upper(),
                "count": query.count(),
                "is_today": current.month == today.month and current.year == today.year
            })
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return results

@router.get("/settings")
def get_settings():
    return config

@router.post("/settings")
def update_settings(settings: dict):
    if save_config(settings):
        if "ai" in settings:
            analysis.init_analysis()
        if "whatsapp" in settings:
            triggers.whatsapp.init_whatsapp()
        if "cameras" in settings:
            triggers.sync_schedules()
        return {"status": "saved"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save config")

@router.get("/whatsapp/status")
def get_whatsapp_status():
    if not triggers.whatsapp.client:
        return {"connected": False, "status": "disabled"}
    
    ok, code = triggers.whatsapp.client.check_connection()
    if ok:
        return {"connected": True, "status": "online"}
    elif code == 401:
        return {"connected": False, "status": "unauthorized"}
    else:
        return {"connected": False, "status": "offline", "code": code}

@router.get("/whatsapp/groups")
def get_whatsapp_groups():
    if not triggers.whatsapp.client:
        return []
    return triggers.whatsapp.client.get_groups()

@router.get("/whatsapp/history/{recipient_id}")
def get_whatsapp_history(recipient_id: str, db: Session = Depends(get_db)):
    from src.database import Notification, Event
    # recipient_id could be the value (phone)
    notifs = db.query(Notification, Event)\
               .filter(Notification.recipient_value == recipient_id)\
               .join(Event, Notification.event_id == Event.id)\
               .order_by(Notification.timestamp.desc())\
               .limit(50).all()
               
    results = []
    for notif, event in notifs:
        results.append({
            "id": notif.id,
            "timestamp": notif.timestamp.isoformat(),
            "status": notif.status,
            "analysis": event.analysis_text,
            "image_url": f"/events/{os.path.basename(event.image_path)}" if event.image_path else ""
        })
    return results

@router.get("/ai/models")
def get_ai_models():
    models = analysis.list_models()
    return {"models": models}

@router.post("/cameras/test")
def test_camera_connection(data: dict):
    source = data.get("source")
    if not source:
        raise HTTPException(status_code=400, detail="Source required")
    
    # ONVIF Handling
    if str(source).startswith("onvif://"):
        try:
            from onvif import ONVIFCamera
        except ImportError:
            return {"status": "fail", "message": "Missing dependency: onvif-zeep. Install it to support ONVIF."}

        try:
            from urllib.parse import urlparse
            parsed = urlparse(source)
            if not parsed.hostname:
                 return {"status": "fail", "message": "Invalid ONVIF URL format. Use onvif://user:pass@ip:port"}

            host = parsed.hostname
            port = parsed.port or 80
            user = parsed.username or 'admin'
            passwd = parsed.password or ''

            logger.info(f"Attempting ONVIF connection to {host}:{port} as {user}")

            # Find WSDL directory (shipped with onvif-zeep package)
            import glob
            wsdl_dir = None
            for candidate in [
                '/usr/local/lib/python3.11/site-packages/wsdl',
                os.path.join(os.path.dirname(__import__('onvif').__file__), 'wsdl'),
            ]:
                if os.path.isdir(candidate):
                    wsdl_dir = candidate
                    break
            
            if not wsdl_dir:
                # Try to find it dynamically
                import site
                for sp in site.getsitepackages():
                    candidate = os.path.join(sp, 'wsdl')
                    if os.path.isdir(candidate):
                        wsdl_dir = candidate
                        break

            if not wsdl_dir:
                return {"status": "fail", "message": "ONVIF WSDL files not found. Reinstall onvif-zeep."}

            logger.info(f"Using WSDL directory: {wsdl_dir}")

            mycam = ONVIFCamera(host, port, user, passwd, wsdl_dir)
            
            # Create media service and get profiles
            media = mycam.create_media_service()
            profiles = media.GetProfiles()
            if not profiles:
                return {"status": "fail", "message": "ONVIF connected but no media profiles found on device."}
            
            token = profiles[0].token
            logger.info(f"ONVIF: Using profile '{token}', found {len(profiles)} profile(s)")
            
            # Get Stream URI
            obj = media.create_type('GetStreamUri')
            obj.StreamSetup = {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}
            obj.ProfileToken = token
            
            res = media.GetStreamUri(obj)
            rtsp_uri = res.Uri
            logger.info(f"ONVIF resolved RTSP URI: {rtsp_uri}")

            # Inject credentials into RTSP URI if missing
            if user and passwd and "://" in rtsp_uri:
                proto, rest = rtsp_uri.split("://", 1)
                if "@" not in rest:
                    rtsp_uri = f"{proto}://{user}:{passwd}@{rest}"

            return {"status": "ok", "message": f"ONVIF OK! Stream: {rtsp_uri}", "resolved_source": rtsp_uri}

        except Exception as e:
            logger.error(f"ONVIF Error: {e}", exc_info=True)
            return {"status": "fail", "message": f"ONVIF Failed: {str(e)}"}

    try:
        cap_source = int(source) if str(source).isdigit() else source
        logger.info(f"Testing camera connection: {cap_source}")
        
        # Set environment variable for RTSP over TCP (more reliable)
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

        cap = cv2.VideoCapture(cap_source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)
        
        if cap.isOpened():
            success, _ = cap.read()
            cap.release()
            if success:
                return {"status": "ok", "message": "Connection successful"}
            return {"status": "fail", "message": "Connected but no frame received. Check credentials or stream path."}
        
        cap.release()
        return {"status": "fail", "message": "Connection failed. Check IP, port, path, and credentials."}
    except Exception as e:
        logger.error(f"Camera test error: {e}")
        return {"status": "error", "message": str(e)}

@router.post("/cameras")
def update_camera(cam_data: dict):
    cam_id = cam_data.get("id")
    if not cam_id:
        raise HTTPException(status_code=400, detail="Camera ID required")
    
    cam_id = "".join(x for x in cam_id if x.isalnum() or x in "-_")
    current_cameras = config.get("cameras", {})
    
    config_entry = {
        "name": cam_data.get("name", "Camera"),
        "source": cam_data.get("source", 0),
        "analysis_prompt": cam_data.get("prompt", ""),
        "message_instruction": cam_data.get("message_instruction", ""),
        "whatsapp_trigger_phrase": cam_data.get("whatsapp_trigger_phrase", ""),
        "webhook_enabled": cam_data.get("webhook_enabled", False),
        "webhook_url": cam_data.get("webhook_url", ""),
        "schedule_enabled": cam_data.get("schedule_enabled", False),
        "schedule_interval_hrs": int(cam_data.get("schedule_interval_hrs", 1)),
        "schedule_interval_mins": int(cam_data.get("schedule_interval_mins", 0)),
        "mqtt_enabled": cam_data.get("mqtt_enabled", False),
        "mqtt_topic": cam_data.get("mqtt_topic", ""),
        "on_dashboard": cam_data.get("on_dashboard", True),
        "enabled": True
    }
    
    current_cameras[cam_id] = config_entry
    
    if save_config({"cameras": current_cameras}):
        camera_manager.add_camera(cam_id, config_entry["source"], name=config_entry["name"])
        triggers.sync_schedules()
        logger.info(f"Updated camera {cam_id}")
        return {"status": "updated", "id": cam_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to save config")

@router.delete("/cameras/{camera_id}")
def delete_camera(camera_id: str):
    current_cameras = config.get("cameras", {})
    if camera_id in current_cameras:
        del current_cameras[camera_id]
        if save_config({"cameras": current_cameras}):
            camera_manager.remove_camera(camera_id)
            triggers.sync_schedules()
            return {"status": "deleted", "id": camera_id}
    raise HTTPException(status_code=404, detail="Camera not found in config")

@router.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        logger.info(f"WhatsApp Webhook received: {data}")
        
        # Heuristic extraction of text and sender
        text = ""
        sender = ""
        
        # Structure assumption based on common WAHA / GOWA / Multidevice
        # data might contain 'data' object wrapper or be flat
        payload = data
        if "data" in data and isinstance(data["data"], dict):
            payload = data["data"]

        # Extract message text
        if "message" in payload:
            msg_obj = payload["message"]
            if isinstance(msg_obj, dict):
                text = msg_obj.get("conversation") or \
                       msg_obj.get("extendedTextMessage", {}).get("text") or \
                       msg_obj.get("imageMessage", {}).get("caption") or ""
            elif isinstance(msg_obj, str):
                text = msg_obj
        
        # Fallback for other formats
        if not text and "text" in payload:
            if isinstance(payload["text"], str):
                text = payload["text"]
            elif isinstance(payload["text"], dict):
                text = payload["text"].get("body", "")
            
        # Extract sender
        if "key" in payload and "remoteJid" in payload["key"]:
            sender = payload["key"]["remoteJid"]
        elif "from" in payload:
            sender = payload["from"]
            
        if user_id := payload.get("participant"): # Group handling
            sender = user_id

        if not text or not sender:
            # Maybe it's a different structure? Just log and ignore
            return {"status": "ignored", "reason": "no_text_or_sender"}
            
        if "@" in sender:
            participant_number = sender.split("@")[0]
        else:
            participant_number = sender

        logger.info(f"WA Message from {participant_number}: {text}")
        

        # Check Patrol Trigger
        patrol_config = config.get("patrol", {})
        patrol_trigger = patrol_config.get("whatsapp_trigger_phrase", "").strip()
        
        # Priority 1: Home Patrol
        if patrol_trigger and patrol_trigger.lower() in text.lower():
            logger.info("Matched Home Patrol Trigger")
            recipients = patrol_config.get("recipients") or config.get("whatsapp", {}).get("recipients", [])
            if is_authorized(sender, recipients):
                logger.info(f"Triggering Home Patrol via WhatsApp from {participant_number}")
                background_tasks.add_task(triggers.perform_home_patrol)
                return {"status": "triggered", "action": "home_patrol"}
            else:
                logger.warning(f"Unauthorized Home Patrol attempt from {participant_number}")
                return {"status": "unauthorized"}

        # Priority 2: Individual Camera Triggers
        if "check" in text.lower() or "look" in text.lower() or "liat" in text.lower(): # Optimization: only check cams if some intent keyword is present, or just always check
            # Actually, user might set any trigger phrase, so we should check all
            pass

        cameras = config.get("cameras", {})
        for cam_id, cam_conf in cameras.items():
            cam_trigger = cam_conf.get("whatsapp_trigger_phrase", "").strip()
            if cam_trigger and cam_trigger.lower() in text.lower():
                logger.info(f"Matched Camera Trigger for {cam_id}")
                recipients = cam_conf.get("recipients") or config.get("whatsapp", {}).get("recipients", [])
                if is_authorized(sender, recipients):
                    logger.info(f"Triggering analysis for {cam_id} via WhatsApp from {participant_number}")
                    background_tasks.add_task(triggers.perform_analysis, cam_id)
                    return {"status": "triggered", "action": "camera_analysis", "camera": cam_id}
                else:
                    logger.warning(f"Unauthorized camera trigger attempt for {cam_id} from {participant_number}")
                    return {"status": "unauthorized"}



        return {"status": "processed", "message": "No trigger matched"}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "detail": str(e)}

def is_authorized(sender, recipients_raw):
    allowed_numbers = []
    
    # Use same logic as whatsapp.py to parse recipients
    recipients_list = []
    if isinstance(recipients_raw, str):
        try:
            import json
            parsed = json.loads(recipients_raw)
            if isinstance(parsed, list):
                recipients_list = parsed
            else:
                recipients_list = [{"name": r.strip(), "value": r.strip()} for r in recipients_raw.replace(',', ' ').split() if r.strip()]
        except:
             recipients_list = [{"name": r.strip(), "value": r.strip()} for r in recipients_raw.replace(',', ' ').split() if r.strip()]
    elif isinstance(recipients_raw, list):
        for r in recipients_raw:
            if isinstance(r, dict):
                recipients_list.append(r)
            elif isinstance(r, str):
                recipients_list.append({"name": r, "value": r})

    for r in recipients_list:
        val = r.get("value", "")
        allowed_numbers.append("".join(filter(str.isdigit, str(val))))
    
    sender_norm = "".join(filter(str.isdigit, str(sender)))
    if "@" in sender_norm: sender_norm = sender_norm.split("@")[0]

    for allowed in allowed_numbers:
        if allowed and allowed in sender_norm: 
            return True
    return False

# ─── Backup & Restore ────────────────────────────────────────────────────────

DATA_DIR = "/data"

@router.get("/backup/info")
def backup_info():
    """Return inventory of what would be backed up."""
    cameras_count = len(config.get("cameras", {}))
    ai_provider = config.get("ai", {}).get("provider", "None")
    
    # Count faces
    faces_dir = os.path.join(DATA_DIR, "faces")
    faces_count = 0
    if os.path.isdir(faces_dir):
        faces_count = len([d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))])
    
    # Count events
    events_count = 0
    try:
        db = SessionLocal()
        events_count = db.query(Event).count()
        db.close()
    except:
        pass
    
    return {
        "cameras": cameras_count,
        "ai_provider": ai_provider.upper() if ai_provider else "None",
        "faces": faces_count,
        "events": events_count,
    }

@router.get("/backup/create")
def create_backup():
    """Create a ZIP backup of the entire /data directory."""
    import zipfile
    import io
    from datetime import datetime
    from fastapi.responses import Response
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"velovision_backup_{timestamp}.zip"
    
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, DATA_DIR)
                try:
                    zf.write(filepath, arcname)
                except Exception as e:
                    logger.warning(f"Skipping {filepath}: {e}")
    
    buffer.seek(0)
    
    logger.info(f"Backup created: {filename} ({buffer.getbuffer().nbytes} bytes)")
    
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

@router.post("/backup/restore")
async def restore_backup(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Restore a ZIP backup to the /data directory."""
    import zipfile
    import io
    from src.config import reload_config
    
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted")
    
    try:
        # Save the uploaded file to a temporary location first
        temp_zip = "/tmp/restore_temp.zip"
        with open(temp_zip, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # We handle the entire extraction and re-initialization in a background task
        def perform_full_restore():
            try:
                import time
                time.sleep(2) # Give the response time to finish
                logger.info("Starting background restore process...")
                
                # 1. Stop all cameras
                for cam_id in list(camera_manager.cameras.keys()):
                    try:
                        camera_manager.remove_camera(cam_id)
                    except:
                        pass
                
                # 2. Stop scheduler
                try:
                    triggers.scheduler.remove_all_jobs()
                except:
                    pass

                # 3. Dispose DB to release file locks
                try:
                    engine.dispose()
                except:
                    pass

                # 4. Extract Zip
                import zipfile
                with zipfile.ZipFile(temp_zip, 'r') as zf:
                    for member in zf.infolist():
                        try:
                            zf.extract(member, DATA_DIR)
                        except Exception as e:
                            logger.error(f"Extraction error ({member.filename}): {e}")

                # 5. Reload system
                new_config = reload_config()
                from src.database import init_db
                init_db()

                for cam_id, cam_cfg in new_config.get("cameras", {}).items():
                    if cam_cfg.get("enabled", True):
                        source = cam_cfg.get("source", 0)
                        try:
                            camera_manager.add_camera(cam_id, source, name=cam_cfg.get("name", cam_id))
                        except:
                            pass
                
                triggers.sync_schedules()
                analysis.init_analysis()
                whatsapp.init_whatsapp()
                
                # 6. Cleanup
                if os.path.exists(temp_zip):
                    os.remove(temp_zip)
                
                logger.info("BACKGROUND RESTORE COMPLETE.")
            except Exception as bge:
                logger.error(f"Background restore failed: {bge}", exc_info=True)

        background_tasks.add_task(perform_full_restore)
        
        logger.info(f"Restore request accepted for {file.filename}")
        return {"status": "ok", "message": "Restore started in background. Page will refresh shortly."}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Restore failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")
