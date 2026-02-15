from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Request, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, FileResponse
from src.streaming import camera_manager, generate_frames
from src.config import config, save_config
from src import analysis, triggers, whatsapp, mqtt
from sqlalchemy.orm import Session
from src.database import get_db, Event, Face, SessionLocal, UnlabeledPerson, engine, ActionLog
import shutil
import logging
import os
import json
import cv2

logger = logging.getLogger(__name__)
router = APIRouter()

# Camera initialization moved to startup event (triggers.start_scheduler handles it)
_cameras_initialized = False

def _ensure_cameras_initialized():
    """Initialize cameras from config. Safe to call multiple times."""
    global _cameras_initialized
    if _cameras_initialized:
        return
    _cameras_initialized = True
    
    for cam_id, cam_cfg in config.get("cameras", {}).items():
        if cam_cfg.get("enabled", True):
            source = cam_cfg.get("source", 0)
            name = cam_cfg.get("name", cam_id)
            logger.info(f"Initializing camera: {cam_id} (source: {source})")
            try:
                camera_manager.add_camera(cam_id, source, name=name)
            except Exception as e:
                logger.error(f"Failed to add camera {cam_id}: {e}")

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

@router.get("/action_logs")
def get_action_logs(limit: int = 20, offset: int = 0, action_type: str = None, db: Session = Depends(get_db)):
    query = db.query(ActionLog)
    if action_type:
        query = query.filter(ActionLog.action_type == action_type)
    
    logs = query.order_by(ActionLog.timestamp.desc()).limit(limit).offset(offset).all()
    results = []
    for log in logs:
        results.append({
            "id": log.id,
            "type": log.action_type, # "home_patrol", "person_finder"
            "timestamp": log.timestamp.isoformat(),
            "summary": log.summary,
            "details": json.loads(log.details) if log.details else {},
            "image_url": log.image_path or ""
        })
    return results

@router.get("/faces")
def get_faces(db: Session = Depends(get_db)):
    if not analysis.face_manager:
        raise HTTPException(status_code=503, detail="Face Manager not initialized")
    faces_records = db.query(Face).all()
    results = []
    for face in faces_records:
        results.append({
            "id": face.id,
            "name": face.name,
            "category": face.category or "Uncategorized",
            "last_seen": face.last_seen.isoformat() if face.last_seen else None,
            "sighting_count": face.sighting_count
        })
    return {"faces": results}

@router.get("/faces/{name}/image")
def get_face_image(name: str, db: Session = Depends(get_db)):
    face = db.query(Face).filter(Face.name == name).first()
    path = None
    if face and face.image_path and os.path.exists(face.image_path):
        path = face.image_path
    else:
        # Fallback disk check
        for ext in ['.jpg', '.jpeg', '.png']:
            p = os.path.join("/data/faces", f"{name}{ext}")
            if os.path.exists(p):
                path = p
                break
    
    if not path:
        raise HTTPException(status_code=404, detail="Face image not found")
    return FileResponse(path)

@router.get("/faces/{name}/sightings")
def get_face_sightings(name: str, db: Session = Depends(get_db)):
    events = db.query(Event).filter(Event.faces_detected.contains(name)).order_by(Event.timestamp.desc()).all()
    return [{
        "id": e.id,
        "timestamp": e.timestamp.isoformat(),
        "camera_id": e.camera_id,
        "image": f"/events/{os.path.basename(e.image_path)}" if e.image_path else None
    } for e in events]

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
        # 1. Update existing face
        existing.last_seen = person.timestamp
        existing.sighting_count = (existing.sighting_count or 0) + 1
        if category and category != "Uncategorized":
            existing.category = category
            
        # 2. Update the source event
        if person.event_id:
            from src.database import Event
            event = db.query(Event).filter(Event.id == person.event_id).first()
            if event:
                # Append to faces if others already detected? 
                if event.faces_detected:
                    names = event.faces_detected.split(",")
                    if name not in names:
                        names.append(name)
                        event.faces_detected = ",".join(names)
                else:
                    event.faces_detected = name
        
        # 3. Cleanup
        db.delete(person)
        db.commit()
        return {"status": "success", "message": f"Added to {name}'s history"}
    
    # Capture the image from the unlabeled entry
    if not os.path.exists(person.image_path):
         raise HTTPException(status_code=404, detail="Reference image missing")
    
    with open(person.image_path, "rb") as f:
        content = f.read()
    
    # 1. Save the image to the faces directory (always succeeds)
    faces_dir = "/data/faces"
    os.makedirs(faces_dir, exist_ok=True)
    face_image_path = os.path.join(faces_dir, f"{name}.jpg")
    with open(face_image_path, "wb") as f:
        f.write(content)
    
    # 2. Register in FaceManager's known list (lightweight, no face_recognition needed)
    #    The image is already saved to disk above. Gemini handles recognition via reference images.
    try:
        if analysis.face_manager:
            if name not in analysis.face_manager.known_face_names:
                analysis.face_manager.known_face_names.append(name)
                logger.info(f"Registered face name: {name}")
    except Exception as e:
        logger.warning(f"Face registration error for {name}: {e}")
    
    # 3. Add to database
    new_face = Face(
        name=name,
        category=category,
        image_path=face_image_path,
        last_seen=person.timestamp,
        sighting_count=0
    )
    
    # Update the source event so it shows up in sightings gallery immediately
    if person.event_id:
        from src.database import Event
        event = db.query(Event).filter(Event.id == person.event_id).first()
        if event:
            event.faces_detected = name
            new_face.sighting_count = 1
            
    db.add(new_face)
    
    # 4. Cleanup: Remove the unlabeled record
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
    latest_log = db.query(ActionLog).order_by(ActionLog.timestamp.desc()).first()
    
    latest_event_data = None
    
    # Determine which is latest
    target = None
    if latest_event and latest_log:
        if latest_log.timestamp > latest_event.timestamp:
            target = ('log', latest_log)
        else:
            target = ('event', latest_event)
    elif latest_event:
        target = ('event', latest_event)
    elif latest_log:
        target = ('log', latest_log)
        
    if target:
        type_, item = target
        if type_ == 'event':
            image_url = f"/events/{os.path.basename(item.image_path)}" if item.image_path else None
            faces = [f.strip() for f in item.faces_detected.split(',')] if item.faces_detected else []
            latest_event_data = {
                "id": item.id,
                "timestamp": item.timestamp.isoformat(),
                "image_url": image_url,
                "faces": faces,
                "camera_id": item.camera_id,
                "analysis_text": item.analysis_text or ""
            }
        else:
            # ActionLog stores the web path directly in image_path
            latest_event_data = {
                "id": item.id,
                "timestamp": item.timestamp.isoformat(),
                "image_url": item.image_path,
                "faces": [item.action_type.replace('_', ' ').title()],
                "camera_id": "System Action",
                "analysis_text": item.summary or ""
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

@router.get("/analytics/summary")
def get_analytics_summary(db: Session = Depends(get_db)):
    from sqlalchemy import func
    from datetime import datetime, timedelta
    import os, shutil

    # 1. Hourly activity for the last 24 hours
    now = datetime.now()
    hourly_activity = []
    for i in range(23, -1, -1):
        target_time = (now - timedelta(hours=i))
        hour_start = target_time.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)
        count = db.query(Event).filter(Event.timestamp >= hour_start, Event.timestamp < hour_end).count()
        hourly_activity.append({
            "label": hour_start.strftime("%I%p"),
            "count": count
        })

    # 2. Events by Camera
    camera_stats = db.query(Event.camera_id, func.count(Event.id)).group_by(Event.camera_id).all()
    cam_breakdown = []
    total_events = sum(c[1] for c in camera_stats) or 1
    for cam_id, count in camera_stats:
        cam_breakdown.append({
            "id": cam_id,
            "count": count,
            "percent": round((count / total_events) * 100)
        })

    # 3. Top Faces
    face_stats = db.query(Face.name, Face.sighting_count).order_by(Face.sighting_count.desc()).limit(5).all()
    top_faces = [{"name": f[0], "count": f[1]} for f in face_stats]

    # 4. System health
    try:
        # CPU Load
        load = os.getloadavg()[0] * 10 
        cpu = round(min(load, 100.0), 1)
        
        # Disk usage
        du = shutil.disk_usage("/")
        storage_percent = round((du.used / du.total) * 100)
        
        # Memory usage (Fallback for Linux)
        mem_percent = 45 
        if os.path.exists("/proc/meminfo"):
             with open("/proc/meminfo", "r") as f:
                 lines = f.readlines()
                 total = 0
                 avail = 0
                 for line in lines:
                     if "MemTotal" in line: total = int(line.split()[1])
                     if "MemAvailable" in line: avail = int(line.split()[1])
                 if total > 0:
                     mem_percent = round(((total - avail) / total) * 100)
    except:
        cpu, storage_percent, mem_percent = 0, 0, 0

    # 5. Common Objects (parsing last 50 events)
    objects = {}
    recent_events = db.query(Event.analysis_text).order_by(Event.id.desc()).limit(50).all()
    keywords = ["person", "car", "package", "dog", "cat", "bicycle", "motorcycle", "delivery"]
    for (text,) in recent_events:
        if not text: continue
        t = text.lower()
        for k in keywords:
            if k in t:
                objects[k] = objects.get(k, 0) + 1
    
    obj_list = [{"label": k.capitalize(), "count": v} for k, v in objects.items()]
    obj_list = sorted(obj_list, key=lambda x: x["count"], reverse=True)

    return {
        "hourly": hourly_activity,
        "cameras": cam_breakdown,
        "faces": top_faces,
        "objects": obj_list,
        "health": {
            "cpu": cpu,
            "storage": storage_percent,
            "memory": mem_percent
        }
    }

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
        if any(k in settings for k in ["cameras", "person_finder", "utility_meters", "patrol", "doorbell_iq"]):
            triggers.sync_schedules()
            # Refresh MQTT discovery if devices changed
            if mqtt.client and mqtt.client.connected:
                mqtt.client.publish_discovery()
        if "mqtt" in settings:
            mqtt.init_mqtt()
        return {"status": "saved"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save config")

@router.get("/mqtt/status")
def get_mqtt_status():
    if not mqtt.client:
        return {"available": mqtt.MQTT_AVAILABLE, "connected": False, "status": "disabled"}
    return mqtt.client.get_status()

@router.get("/mqtt/topics")
def get_mqtt_topics():
    return mqtt.get_topics_info()

@router.get("/mqtt/messages")
def get_mqtt_messages():
    if not mqtt.client:
        return []
    return mqtt.client.get_recent_messages()

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
            sender = payload["key"].get("remoteJid")
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
        

        # Individual Camera Triggers (Intent-based)
        if "check" in text.lower() or "look" in text.lower() or "liat" in text.lower():
             # Basic intent-based triggering logic can be added here if needed in future
             pass

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
    
    # Count faces & Discoveries
    try:
        db = SessionLocal()
        faces_count = db.query(Face).count()
        unlabeled_count = db.query(UnlabeledPerson).count()
        events_count = db.query(Event).count()
        db.close()
    except:
        faces_count = 0
        unlabeled_count = 0
        events_count = 0
    
    return {
        "cameras": cameras_count,
        "ai_provider": ai_provider.upper() if ai_provider else "None",
        "faces": faces_count,
        "discoveries": unlabeled_count,
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
                    import gc
                    gc.collect() # Force cleanup of connections
                except:
                    pass

                # 4. Extract Zip
                import zipfile
                # Ensure target directories exist before extraction
                if not os.path.exists(DATA_DIR):
                     os.makedirs(DATA_DIR, exist_ok=True)
                
                with zipfile.ZipFile(temp_zip, 'r') as zf:
                    for member in zf.infolist():
                        try:
                            # Sanitize paths
                            if '..' in member.filename: continue
                            if member.filename.startswith('__MACOSX') or '/.' in member.filename or member.filename.startswith('.'):
                                continue
                                
                            zf.extract(member, DATA_DIR)
                        except Exception as e:
                            logger.error(f"Extraction error ({member.filename}): {e}")

                # 5. Clean Exit (Let Docker Restart)
                logger.info("Backup restored. Triggering application restart...")
                
                # Small delay to ensure logs are flushed
                import time
                time.sleep(1)
                
                # Cleanup temp file
                if os.path.exists(temp_zip):
                    try:
                        os.remove(temp_zip)
                    except: pass

                # Exit the process. With restart: unless-stopped in Docker, this will reboot the app cleanly.
                os._exit(0)

            except Exception as bge:
                logger.error(f"Background restore failed: {bge}", exc_info=True)

        background_tasks.add_task(perform_full_restore)
        
        logger.info(f"Restore request accepted for {file.filename}")
        return {"status": "ok", "message": "Restore in progress. Application will restart in a few seconds."}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Restore failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")
