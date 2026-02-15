import json
import logging
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt_lib
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed. MQTT features disabled.")

# All available MQTT topics with descriptions
TOPICS = {
    "velovision/status": {
        "description": "System online/offline status (LWT)",
        "type": "status",
        "payload_example": {"status": "online", "version": "1.5.6", "uptime_seconds": 12345},
        "ha_discovery": "binary_sensor"
    },
    "velovision/cameras/{camera_id}/event": {
        "description": "Per-camera AI analysis event with detection details",
        "type": "event",
        "payload_example": {
            "camera_id": "front_door",
            "camera_name": "Front Door",
            "timestamp": "2026-02-15T05:00:00",
            "summary": "Person detected at front door",
            "persons_detected": ["Alice", "Unknown"],
            "person_count": 2,
            "has_motion": True
        },
        "ha_discovery": "sensor"
    },
    "velovision/cameras/{camera_id}/snapshot": {
        "description": "Latest camera snapshot (base64 JPEG, published on analysis)",
        "type": "image",
        "payload_example": "<base64 encoded JPEG>",
        "ha_discovery": "camera"
    },
    "velovision/patrol/result": {
        "description": "Home patrol summary with detected persons and primary camera",
        "type": "event",
        "payload_example": {
            "timestamp": "2026-02-15T05:00:00",
            "summary": "All clear. No unusual activity detected.",
            "primary_camera": "backyard",
            "recognized": ["Alice"],
            "unknown_count": 0,
            "cameras_scanned": 4
        },
        "ha_discovery": "sensor"
    },
    "velovision/person_finder/result": {
        "description": "Person finder scan results with locations and activities",
        "type": "event",
        "payload_example": {
            "timestamp": "2026-02-15T05:00:00",
            "targets": ["Alice", "Bob"],
            "found": {"Alice": [{"camera_name": "Living Room", "activity": "Watching TV", "confidence": "high"}]},
            "not_found": ["Bob"],
            "cameras_scanned": 4
        },
        "ha_discovery": "sensor"
    },
    "velovision/faces/detected": {
        "description": "New or known face detection event",
        "type": "event",
        "payload_example": {
            "timestamp": "2026-02-15T05:00:00",
            "camera_id": "front_door",
            "name": "Alice",
            "category": "Family",
            "is_new": False
        },
        "ha_discovery": "sensor"
    },
    "velovision/trigger/patrol": {
        "description": "Trigger a home patrol scan (subscribe to this topic)",
        "type": "trigger",
        "payload_example": {"action": "run"},
        "direction": "inbound"
    },
    "velovision/trigger/person_finder": {
        "description": "Trigger a person finder scan (subscribe to this topic)",
        "type": "trigger",
        "payload_example": {"action": "run", "names": ["Alice", "Bob"]},
        "direction": "inbound"
    },
    "velovision/trigger/analyze/{camera_id}": {
        "description": "Trigger AI analysis on a specific camera (subscribe)",
        "type": "trigger",
        "payload_example": {"action": "run"},
        "direction": "inbound"
    },
    "velovision/trigger/doorbell_iq": {
        "description": "Trigger Doorbell IQ analysis (subscribe)",
        "type": "trigger",
        "payload_example": {"action": "run"},
        "direction": "inbound"
    },
    "velovision/trigger/utility_meter": {
        "description": "Trigger Utility Meter reading (subscribe)",
        "type": "trigger",
        "payload_example": {"action": "run"},
        "direction": "inbound"
    }
}


class MQTTClient:
    def __init__(self, broker_host, broker_port=1883, username=None, password=None,
                 client_id="velovision", base_topic="velovision"):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.client_id = client_id
        self.base_topic = base_topic
        self.connected = False
        self.client = None
        self._lock = threading.Lock()
        self._message_log = []  # Recent published messages for UI

        if not MQTT_AVAILABLE:
            logger.error("paho-mqtt not available, cannot create MQTT client")
            return

        self.client = mqtt_lib.Client(client_id=client_id, protocol=mqtt_lib.MQTTv311)

        if username and password:
            self.client.username_pw_set(username, password)

        # Last Will and Testament
        self.client.will_set(
            f"{base_topic}/status",
            payload=json.dumps({"status": "offline", "timestamp": datetime.now().isoformat()}),
            qos=1,
            retain=True
        )

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def connect(self):
        if not self.client:
            return False
        try:
            logger.info(f"MQTT attempting to connect to {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            # Wait briefly for connection
            time.sleep(1.5)
            if self.connected:
                logger.info(f"MQTT initial connection check: SUCCESS")
            else:
                logger.warning(f"MQTT initial connection check: PENDING (loop is running)")
            return self.connected
        except Exception as e:
            logger.error(f"MQTT Connect failed: {e}")
            return False

    def disconnect(self):
        if self.client and self.connected:
            # Send offline status
            self.publish("status", {"status": "offline", "timestamp": datetime.now().isoformat()}, retain=True)
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            logger.info(f"MQTT Connected to {self.broker_host}:{self.broker_port}")

            # Publish online status
            self.publish("status", {
                "status": "online",
                "timestamp": datetime.now().isoformat(),
                "version": "1.5.6"
            }, retain=True)

            # Subscribe to trigger topics
            client.subscribe(f"{self.base_topic}/trigger/#")
            logger.info("MQTT subscribed to trigger topics")

            # Publish HA Discovery
            self.publish_discovery()
        else:
            logger.error(f"MQTT Connection refused, rc={rc}")
            self.connected = False

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            logger.warning(f"MQTT Unexpected disconnect (rc={rc}), will reconnect")

    def _on_message(self, client, userdata, msg):
        """Handle inbound trigger messages."""
        topic = msg.topic
        try:
            payload = json.loads(msg.payload.decode())
        except:
            payload = msg.payload.decode()

        logger.info(f"MQTT Inbound: {topic} -> {payload}")

        # Route to appropriate handler
        if topic == f"{self.base_topic}/trigger/patrol":
            self._handle_patrol_trigger(payload)
        elif topic == f"{self.base_topic}/trigger/person_finder":
            self._handle_person_finder_trigger(payload)
        elif topic == f"{self.base_topic}/trigger/doorbell_iq":
            self._handle_doorbell_trigger(payload)
        elif topic == f"{self.base_topic}/trigger/utility_meter":
            self._handle_meter_trigger(payload)
        elif topic.startswith(f"{self.base_topic}/trigger/analyze/"):
            cam_id = topic.split("/")[-1]
            self._handle_analysis_trigger(cam_id, payload)

    def _handle_patrol_trigger(self, payload):
        """Trigger home patrol from MQTT."""
        try:
            from src.config import config
            if not config.get("patrol", {}).get("mqtt_enabled"):
                logger.warning("MQTT Home Patrol trigger ignored: mqtt_enabled is false")
                return

            from src import triggers
            threading.Thread(target=triggers.perform_home_patrol, daemon=True).start()
            logger.info("MQTT triggered Home Patrol")
        except Exception as e:
            logger.error(f"MQTT patrol trigger failed: {e}")

    def _handle_person_finder_trigger(self, payload):
        """Trigger person finder from MQTT."""
        try:
            from src.config import config
            if not config.get("person_finder", {}).get("mqtt_enabled"):
                logger.warning("MQTT Person Finder trigger ignored: mqtt_enabled is false")
                return

            from src import triggers
            names = payload.get("names", []) if isinstance(payload, dict) else []
            if names:
                threading.Thread(
                    target=triggers.perform_person_finder,
                    args=(names,),
                    daemon=True
                ).start()
                logger.info(f"MQTT triggered Person Finder for {names}")
        except Exception as e:
            logger.error(f"MQTT person finder trigger failed: {e}")

    def _handle_analysis_trigger(self, camera_id, payload):
        """Trigger single camera analysis from MQTT."""
        try:
            from src.config import config
            cam_cfg = config.get("cameras", {}).get(camera_id)
            if not cam_cfg or not cam_cfg.get("mqtt_enabled"):
                logger.warning(f"MQTT analysis trigger ignored for {camera_id}: mqtt_enabled is false")
                return

            from src import triggers
            threading.Thread(
                target=triggers.perform_analysis,
                args=(camera_id,),
                daemon=True
            ).start()
            logger.info(f"MQTT triggered analysis for {camera_id}")
        except Exception as e:
            logger.error(f"MQTT analysis trigger failed: {e}")

    def _handle_doorbell_trigger(self, payload):
        """Trigger Doorbell IQ from MQTT."""
        try:
            from src.config import config
            if not config.get("doorbell_iq", {}).get("mqtt_enabled"):
                logger.warning("MQTT Doorbell IQ trigger ignored: mqtt_enabled is false")
                return

            from src import triggers
            import threading
            threading.Thread(target=triggers.perform_doorbell_analysis, daemon=True).start()
            logger.info("MQTT triggered Doorbell IQ")
        except Exception as e:
            logger.error(f"MQTT doorbell trigger failed: {e}")

    def _handle_meter_trigger(self, payload):
        """Placeholder for Utility Meter MQTT trigger."""
        logger.info("MQTT Utility Meter trigger received (Coming Soon)")

    def publish(self, topic_suffix, payload, retain=False, qos=0):
        """Publish a message to {base_topic}/{topic_suffix}."""
        if not self.client or not self.connected:
            return False

        full_topic = f"{self.base_topic}/{topic_suffix}"
        try:
            if isinstance(payload, dict):
                msg = json.dumps(payload)
            else:
                msg = str(payload)

            result = self.client.publish(full_topic, msg, qos=qos, retain=retain)

            # Log for UI
            with self._lock:
                self._message_log.append({
                    "topic": full_topic,
                    "payload_preview": msg[:200] if len(msg) > 200 else msg,
                    "timestamp": datetime.now().isoformat(),
                    "retained": retain,
                    "qos": qos
                })
                # Keep only last 50
                if len(self._message_log) > 50:
                    self._message_log = self._message_log[-50:]

            return result.rc == 0
        except Exception as e:
            logger.error(f"MQTT Publish failed ({full_topic}): {e}")
            return False

    def publish_camera_event(self, camera_id, camera_name, summary, persons=None, snapshot_bytes=None):
        """Publish a camera analysis event."""
        event = {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "persons_detected": persons or [],
            "person_count": len(persons) if persons else 0,
            "has_motion": bool(persons)
        }
        self.publish(f"cameras/{camera_id}/event", event)

        # Also publish snapshot if available
        if snapshot_bytes:
            import base64
            b64 = base64.b64encode(snapshot_bytes).decode()
            self.publish(f"cameras/{camera_id}/snapshot", b64)

    def publish_patrol_result(self, summary, primary_camera, recognized, unknown_count, cameras_scanned):
        """Publish home patrol result."""
        self.publish("patrol/result", {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "primary_camera": primary_camera,
            "recognized": recognized,
            "unknown_count": unknown_count,
            "cameras_scanned": cameras_scanned
        })

    def publish_person_finder_result(self, targets, found, not_found, cameras_scanned, summary):
        """Publish person finder results."""
        self.publish("person_finder/result", {
            "timestamp": datetime.now().isoformat(),
            "targets": targets,
            "found": found,
            "not_found": not_found,
            "cameras_scanned": cameras_scanned,
            "summary": summary
        })

    def publish_face_detected(self, camera_id, name, category="Uncategorized", is_new=False):
        """Publish face detection event."""
        self.publish("faces/detected", {
            "timestamp": datetime.now().isoformat(),
            "camera_id": camera_id,
            "name": name,
            "category": category,
            "is_new": is_new
        })

    def publish_discovery(self):
        """Publish Home Assistant MQTT discovery payloads."""
        if not self.client or not self.connected:
            return

        from src.config import config
        disc_prefix = "homeassistant"
        node_id = self.client_id
        
        device_info = {
            "identifiers": [f"velovision_{node_id}"],
            "name": "VeloVision AI Vision",
            "model": "VeloVision Vision Core",
            "manufacturer": "VeloVision",
            "sw_version": "1.5.6"
        }

        # 1. System Status (Binary Sensor)
        self._publish_ha_config(disc_prefix, "binary_sensor", "status", {
            "name": "VeloVision Status",
            "device_class": "connectivity",
            "state_topic": f"{self.base_topic}/status",
            "value_template": "{{ 'ON' if value_json.status == 'online' else 'OFF' }}",
            "payload_on": "ON",
            "payload_off": "OFF",
            "device": device_info,
            "unique_id": f"{node_id}_status"
        })

        # 2. Patrol Summary (Sensor)
        self._publish_ha_config(disc_prefix, "sensor", "patrol_result", {
            "name": "VeloVision Patrol Summary",
            "state_topic": f"{self.base_topic}/patrol/result",
            "value_template": "{{ value_json.summary }}",
            "icon": "mdi:shield-check",
            "json_attributes_topic": f"{self.base_topic}/patrol/result",
            "device": device_info,
            "unique_id": f"{node_id}_patrol_result"
        })

        # 3. Person Finder Summary (Sensor)
        self._publish_ha_config(disc_prefix, "sensor", "person_finder_result", {
            "name": "VeloVision Person Finder",
            "state_topic": f"{self.base_topic}/person_finder/result",
            "value_template": "{{ value_json.summary if value_json.summary else 'Idle' }}",
            "icon": "mdi:account-search",
            "json_attributes_topic": f"{self.base_topic}/person_finder/result",
            "device": device_info,
            "unique_id": f"{node_id}_person_finder_result"
        })

        # 4. Per-Camera Sensors & Cameras
        for cam_id, cam_cfg in config.get("cameras", {}).items():
            if not cam_cfg.get("enabled", True):
                continue
            
            cam_name = cam_cfg.get("name", cam_id)
            clean_id = cam_id.replace("-", "_").replace(" ", "_")

            # Camera Sensor (Last Event)
            self._publish_ha_config(disc_prefix, "sensor", f"{clean_id}_event", {
                "name": f"VeloVision {cam_name} Event",
                "state_topic": f"{self.base_topic}/cameras/{cam_id}/event",
                "value_template": "{{ value_json.summary }}",
                "icon": "mdi:video-account",
                "json_attributes_topic": f"{self.base_topic}/cameras/{cam_id}/event",
                "device": device_info,
                "unique_id": f"{node_id}_{clean_id}_event"
            })

            # Camera Entity (Latest Snapshot)
            # Note: HA MQTT Camera expects raw bytes, but we send base64 currently. 
            # We'll use a snapshot topic for the camera entity.
            self._publish_ha_config(disc_prefix, "camera", f"{clean_id}_snapshot", {
                "name": f"VeloVision {cam_name} Snapshot",
                "topic": f"{self.base_topic}/cameras/{cam_id}/snapshot",
                "device": device_info,
                "unique_id": f"{node_id}_{clean_id}_snapshot"
            })

        logger.info(f"Published HA discovery for {len(config.get('cameras', {}))} cameras and system sensors.")

    def _publish_ha_config(self, prefix, component, object_id, payload):
        """Helper to publish HA MQTT discovery config."""
        topic = f"{prefix}/{component}/{self.client_id}/{object_id}/config"
        self.client.publish(topic, json.dumps(payload), qos=1, retain=True)

    def get_recent_messages(self):
        """Return recent published messages for UI display."""
        with self._lock:
            return list(reversed(self._message_log))

    def get_status(self):
        """Return connection status dict."""
        return {
            "available": MQTT_AVAILABLE,
            "connected": self.connected,
            "broker_host": self.broker_host,
            "broker_port": self.broker_port,
            "client_id": self.client_id,
            "base_topic": self.base_topic,
            "message_count": len(self._message_log)
        }


# Global instance
client = None


def init_mqtt():
    """Initialize MQTT client from config."""
    global client
    from src.config import config

    mqtt_config = config.get("mqtt", {})
    if not mqtt_config.get("enabled"):
        client = None
        logger.info("MQTT is disabled.")
        return

    if not MQTT_AVAILABLE:
        logger.error("MQTT enabled in config but paho-mqtt is not installed!")
        client = None
        return

    broker = mqtt_config.get("broker_host", "localhost")
    port = int(mqtt_config.get("broker_port", 1883))
    user = mqtt_config.get("username", "")
    password = mqtt_config.get("password", "")
    client_id = mqtt_config.get("client_id", "velovision")
    base_topic = mqtt_config.get("base_topic", "velovision")

    logger.info(f"Initializing MQTT client: {broker}:{port} (ID: {client_id}, Topic: {base_topic})")

    # Disconnect existing client
    if client:
        try:
            client.disconnect()
        except:
            pass

    client = MQTTClient(
        broker_host=broker,
        broker_port=port,
        username=user if user else None,
        password=password if password else None,
        client_id=client_id,
        base_topic=base_topic
    )

    client.connect()


def get_topics_info():
    """Return all available topics with descriptions for UI."""
    return TOPICS
