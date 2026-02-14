import requests
import logging
import base64
import io
import json

logger = logging.getLogger(__name__)

class WhatsAppClient:
    def __init__(self, api_url="https://api.gowa.me", session_id="default", username=None, password=None):
        self.api_url = api_url.rstrip("/")
        self.session_id = session_id
        self.username = username
        self.password = password

    def send_alert(self, recipients_raw, image_bytes, caption):
        """
        Sends an image alert to one or more recipients.
        Returns a list of dicts: [{"name": name, "value": val, "success": bool}]
        """
        if not recipients_raw:
            return []

        recipients_list = []
        
        # Try to parse as JSON first
        if isinstance(recipients_raw, str):
            try:
                parsed = json.loads(recipients_raw)
                if isinstance(parsed, list):
                    recipients_list = parsed
                else:
                    recipients_list = [{"name": r.strip(), "value": r.strip()} for r in recipients_raw.replace(',', ' ').split() if r.strip()]
            except json.JSONDecodeError:
                recipients_list = [{"name": r.strip(), "value": r.strip()} for r in recipients_raw.replace(',', ' ').split() if r.strip()]
        elif isinstance(recipients_raw, list):
            for r in recipients_raw:
                if isinstance(r, dict):
                    recipients_list.append(r)
                elif isinstance(r, str):
                    recipients_list.append({"name": r, "value": r})

        if not recipients_list:
            return []

        delivery_results = []
        for r in recipients_list:
            target = r.get('value')
            name = r.get('name', target)
            if not target: continue

            # Normalization for aldinokemal/go-whatsapp-web-multidevice
            chat_id = target
            if "@" not in chat_id:
                if len(target) < 15:
                    chat_id = f"{target}@s.whatsapp.net"
                else:
                    chat_id = f"{target}@g.us"
            
            success = self._send_image_message(chat_id, image_bytes, caption)
            delivery_results.append({"name": name, "value": target, "success": success})
        
        logger.info(f"WhatsApp notifications: {len([r for r in delivery_results if r['success']])}/{len(delivery_results)} sent.")
        return delivery_results

    def _send_image_message(self, chat_id, image_bytes, caption):
        """Internal method to send image via POST /send/image (multipart/form-data)."""
        try:
            url = f"{self.api_url}/send/image"
            
            # This repo expects multipart/form-data
            files = {
                'image': ('detection.jpg', image_bytes, 'image/jpeg')
            }
            data = {
                "phone": chat_id,
                "caption": caption,
                "compress": "true"
            }
            
            headers = {
                "X-Device-Id": self.session_id
            }
            
            auth = (self.username, self.password) if self.username and self.password else None
            
            response = requests.post(url, data=data, files=files, headers=headers, auth=auth, timeout=20)
            
            if response.ok:
                return True
            else:
                logger.error(f"WhatsApp API Error ({response.status_code}): {response.text}")
                return False
        except Exception as e:
            logger.error(f"WhatsApp Connection Error: {e}")
            return False

    def get_groups(self):
        """Fetch list of joined groups from aldinokemal/go-whatsapp-web-multidevice."""
        try:
            url = f"{self.api_url}/user/my/groups"
            headers = {"X-Device-Id": self.session_id}
            auth = (self.username, self.password) if self.username and self.password else None
            
            response = requests.get(url, headers=headers, auth=auth, timeout=20)
            if response.ok:
                res_json = response.json()
                # format is usually {"results": {"data": [...]}}
                results = res_json.get("results", {})
                data = results.get("data", [])
                
                # Normalize for frontend (id, name)
                formatted = []
                for g in data:
                    formatted.append({
                        "id": g.get("JID"),
                        "name": g.get("Name")
                    })
                return formatted
            else:
                logger.error(f"GOWA Group List Error: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Failed to fetch groups: {e}")
            return []

    def check_connection(self):
        """Verify connectivity and credentials using /app/status."""
        try:
            url = f"{self.api_url}/app/status"
            headers = {"X-Device-Id": self.session_id}
            auth = (self.username, self.password) if self.username and self.password else None
            
            response = requests.get(url, headers=headers, auth=auth, timeout=10)
            return response.ok, response.status_code
        except Exception as e:
            logger.error(f"WhatsApp test failed: {e}")
            return False, 0

# Global instance
client = None

def init_whatsapp():
    global client
    from src.config import config
    wa_config = config.get("whatsapp", {})
    if wa_config.get("enabled"):
        url = wa_config.get("api_url", "https://api.gowa.me")
        session = wa_config.get("session_id", "default")
        user = wa_config.get("username")
        password = wa_config.get("password")
        logger.info(f"Initializing WhatsApp Client for GOWA at {url} (Device: {session})")
        client = WhatsAppClient(api_url=url, session_id=session, username=user, password=password)
    else:
        client = None
        logger.info("WhatsApp notifications are disabled.")
