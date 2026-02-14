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
def startup_event():
    # Ensure directories exist
    os.makedirs("/data/faces", exist_ok=True)
    os.makedirs("/data/events", exist_ok=True)
    init_db()

# Mount events if they exist
app.mount("/events", StaticFiles(directory="/data/events"), name="events")

# Mount Routers
app.include_router(api.router, prefix="/api", tags=["API"])
app.include_router(triggers.router, prefix="/api", tags=["Triggers"])
app.include_router(ui.router)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}
