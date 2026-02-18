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
        # Try finding a backup if main is missing
        backup_path = CONFIG_PATH + ".bak"
        if os.path.exists(backup_path):
             logger.warning(f"Main config missing, but found backup at {backup_path}. Restoring...")
             import shutil
             try:
                 shutil.copy2(backup_path, CONFIG_PATH)
             except Exception as e:
                 logger.error(f"Failed to restore backup: {e}")
                 return defaults
        else:
            logger.warning(f"Config file not found at {CONFIG_PATH}. Using defaults.")
            return defaults
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            user_config = yaml.safe_load(f)
            
            # Check for empty or invalid config (corruption)
            if not user_config:
                logger.error(f"Config file at {CONFIG_PATH} is empty or invalid.")
                # Try restoration from backup
                backup_path = CONFIG_PATH + ".bak"
                if os.path.exists(backup_path):
                    logger.warning("Attempting to restore from .bak file...")
                    with open(backup_path, 'r') as bf:
                        backup_config = yaml.safe_load(bf)
                        if backup_config:
                            # Restore successful-ish
                            logger.info("Backup loaded successfully. Using values from backup.")
                            user_config = backup_config
                            # We don't overwrite the corrupt main file immediately to avoid data loss loops, 
                            # but the next save will fix it.
                
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
        
        # Atomic Write Strategy to prevent corruption
        temp_path = CONFIG_PATH + ".tmp"
        backup_path = CONFIG_PATH + ".bak"
        
        # 1. Write to temp file first
        with open(temp_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            f.flush()
            os.fsync(f.fileno()) # Force write to disk
            
        # 2. Key Step: Create/Update backup of current valid config before replacing
        # getting here means temp write succeeded.
        if os.path.exists(CONFIG_PATH):
             import shutil
             try:
                 shutil.copy2(CONFIG_PATH, backup_path)
             except Exception as be:
                 logger.warning(f"Failed to create backup copy: {be}")

        # 3. Atomic Rename (overwrites CONFIG_PATH)
        os.replace(temp_path, CONFIG_PATH)
            
        logger.info("Configuration saved safely.")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False
