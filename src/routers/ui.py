from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="src/templates")

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "active_page": "dashboard"})

@router.get("/timeline", response_class=HTMLResponse)
async def timeline(request: Request):
    return templates.TemplateResponse("timeline.html", {"request": request, "active_page": "timeline"})

@router.get("/faces", response_class=HTMLResponse)
async def faces(request: Request):
    return templates.TemplateResponse("faces.html", {"request": request, "active_page": "faces"})

@router.get("/cameras", response_class=HTMLResponse)
async def cameras(request: Request):
    return templates.TemplateResponse("cameras.html", {"request": request, "active_page": "cameras"})

@router.get("/patrol", response_class=HTMLResponse)
async def patrol(request: Request):
    return templates.TemplateResponse("patrol.html", {"request": request, "active_page": "patrol"})

@router.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request, "active_page": "analytics"})

@router.get("/messages", response_class=HTMLResponse)
async def messages(request: Request):
    return templates.TemplateResponse("messages.html", {"request": request, "active_page": "messages"})

@router.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request, "active_page": "settings"})

# Legacy redirects for .html extensions
@router.get("/index.html")
async def redirect_index():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/")

@router.get("/{full_path}.html")
async def redirect_html(full_path: str):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/{full_path}")
