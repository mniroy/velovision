import yaml
import os
import logging

logger = logging.getLogger(__name__)

CONFIG_PATH = "/data/config.yaml"

def deep_merge(target, source):
    """Recursively merge source into target."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_merge(target[key], value)
        else:
            target[key] = value

def load_config():
    defaults = get_default_config()
    if not os.path.exists(CONFIG_PATH):
        logger.warning(f"Config file not found at {CONFIG_PATH}. Using defaults.")
        return defaults
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            user_config = yaml.safe_load(f)
            if not user_config:
                return defaults
            # Ensure all default keys exist
            deep_merge(defaults, user_config)
            logger.info(f"Loaded config from {CONFIG_PATH}. MQTT Enabled: {defaults.get('mqtt', {}).get('enabled')}")
            return defaults
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return defaults

def get_default_config():
    return {
        "cameras": {},
        "ai": {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "language": "Bahasa Indonesia"
        },
        "whatsapp": {
            "enabled": False,
            "api_url": "https://api.gowa.me",
            "api_key": "",
            "username": "",
            "password": "",
            "recipients": "[]",
            "session_id": "default"
        },
        "patrol": {
            "prompt": "Perform a holistic security patrol of the entire property using these camera snapshots. Summarize the state of the home. If everything is normal, say 'All Clear'. If there are any anomalies or people detected, describe them clearly.",
            "message_instruction": "",
            "whatsapp_trigger_phrase": "",
            "recipients": [],
            "webhook_enabled": False,
            "mqtt_enabled": False,
            "schedule_enabled": False,
            "schedule_interval_hrs": 6
        },
        "person_finder": {
            "names": [],
            "prompt": "",
            "recipients": [],
            "webhook_enabled": False,
            "mqtt_enabled": False,
            "schedule_enabled": False,
            "schedule_interval_hrs": 4
        },
        "doorbell_iq": {
            "camera_id": "",
            "webhook_enabled": False,
            "mqtt_enabled": False,
            "analysis_prompt": "Analyse who is at the door, their appearance, and any objects they are carrying (e.g. packages). Determine if they look like a delivery person, friend, or stranger.",
            "message_prompt": "Keep it brief and professional. Start with 'Doorbell Alert:'.",
            "include_image": True,
            "recipients_whatsapp": [],
            "recipients_webhook": "",
            "recipients_mqtt": ""
        },
        "utility_meters": [],
        "mqtt": {
            "enabled": False,
            "broker_host": "localhost",
            "broker_port": 1883,
            "username": "",
            "password": "",
            "client_id": "velovision",
            "base_topic": "velovision"
        },
        "general": {
            "timezone": "Asia/Jakarta"
        }
    }

# Global config instance
config = load_config()

def apply_timezone():
    """Apply the configured timezone to the process."""
    tz = config.get("general", {}).get("timezone", "Asia/Jakarta")
    import os
    import time
    os.environ['TZ'] = tz
    try:
        time.tzset()
        logger.info(f"System timezone successfully set to: {tz}")
    except AttributeError:
        # Windows doesn't have tzset
        logger.warning("time.tzset() not available on this platform.")

# Initial apply
apply_timezone()

def reload_config():
    global config
    new_data = load_config()
    # Update the existing dictionary in-place to preserve references in other modules
    config.clear()
    config.update(new_data)
    apply_timezone() # Re-apply on reload
    logger.info("Configuration reloaded in-place.")
    return config

def save_config(updates):
    """Update config with new values and save to YAML."""
    global config
    try:
        # Use deep merge to avoid wiping nested keys (like whatsapp secret vs recipients)
        deep_merge(config, updates)
        
        # Apply any general setting changes (like timezone)
        apply_timezone()
        
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info("Configuration saved.")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False
