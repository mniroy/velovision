from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from src.routers import ui, api
from src import triggers
from src.database import init_db

app = FastAPI(title="Velo Vision")

@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        
        # Initialize heavy components in a background thread
        import threading
        def background_init():
            try:
                from src import triggers, mqtt
                triggers.start_scheduler()
                mqtt.init_mqtt()
            except Exception as thread_e:
                print(f"BACKGROUND INIT ERROR: {thread_e}")
        
        threading.Thread(target=background_init, daemon=True).start()
        
    except Exception as e:
        print(f"CRITICAL STARTUP ERROR: {e}")
# Ensure directories exist at module level so StaticFiles can mount them securely checks
try:
    os.makedirs("/data/faces", exist_ok=True)
    os.makedirs("/data/events", exist_ok=True)
except Exception as e:
    print(f"Directory creation error: {e}")

# Mount static directories
app.mount("/static", StaticFiles(directory="src/static"), name="static")
app.mount("/events", StaticFiles(directory="/data/events"), name="events")

# Mount Routers
app.include_router(api.router, prefix="/api", tags=["API"])
app.include_router(triggers.router, prefix="/api", tags=["Triggers"])
app.include_router(ui.router)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}
