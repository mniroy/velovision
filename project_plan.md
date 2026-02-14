# Velo Vision AI Analyzer - Project Plan

## 1. Project Overview
**Velo Vision** is a containerized application acting as an intelligent middleware for comprehensive home surveillance and camera analysis. It connects to camera RTSP streams to provide a **Live Video Preview** and performs advanced AI analysis for various scenarios (Doorbell, Room Inspection, Security, etc.) when triggered by various events.

## 2. Architecture

### Core Functionality
1.  **Live Monitoring**:
    *   The application connects to multiple configured RTSP streams.
    *   It provides a **Live Video Preview** (MJPEG Stream) accessible via the Web UI.
2.  **Event Trigger (Analysis)**:
    *   **Triggers**:
        *   **Webhook**: `POST /api/trigger?camera=front_door` (e.g., from Home Assistant).
        *   **Schedule**: run analysis at specific times (e.g., "Check back door every night at 11 PM").
        *   **Message**: particular keyword or payload sent to an endpoint or via MQTT.
    *   **Capture**: The system immediately captures the current frame from the active stream.
    *   **Processing**:
        *   **Face Recognition**: Identifies known individuals from the `/data/faces` directory.
        *   **Context Analysis**: Sends the frame + identified names to an AI provider (OpenAI/Gemini/Ollama). The prompt is **camera-specific**.
    *   **Notification**: Sends the analysis result (image + text) to WhatsApp via **go-whatsapp-web-multidevice**.
        *   Supports multiple WhatsApp accounts (devices).
        *   Each recipient can be assigned a specific sending `device_id`.
    *   **timeline**: Saves the event to the local database for historical review.

### Docker Structure
A single container solution.

*   **App Service**: Python (FastAPI).
*   **Database**: SQLite (embedded).
*   **Volumes**:
    *   `/data`: Stores faces, event images, database, and config.

## 3. Tech Stack
*   **Language**: Python 3.10+
*   **Web Framework**: FastAPI + APScheduler (for scheduled triggers).
*   **Video Processing**: OpenCV (`cv2`).
*   **Computer Vision**: `face_recognition` (dlib).
*   **AI Analysis**: OpenAI / Gemini / Ollama.
*   **Database**: SQLite + SQLAlchemy.
*   **Frontend**: HTML/TailwindCSS + Jinja2 Templates.

## 4. Key Features

| Feature | Description |
| :--- | :--- |
| **Live Preview** | Always-on video feed in the Web UI. |
| **RTSP Integration** | Connects to standard RTSP streams. |
| **Camera-Specific AI** | Each camera has its own unique system prompt (Analysis & Notification instructions). |
| **Multi-Trigger** | Trigger via Webhook, Schedule (Interval), or External Message. |
| **Face ID** | Learn and identify faces. |
| **AI Analysis** | Generative AI description of the scene/event. |
| **Timeline Review** | Browse past events. |
| **Analytics** | Visitor stats and system usage. |
| **AI Settings** | Manage AI Providers (OpenAI/Gemini/Ollama) and view Token Usage. |
| **WhatsApp** | Instant alerts via GOWA. |

## 5. Development Phases

### Phase 1: Stream & UI Foundation
*   [x] Setup Dockerfile (Python, OpenCV, Face Rec).
*   [x] Implement **MJPEG Streaming**.
*   [x] Build the Web UI shell.

### Phase 2: Analysis Pipeline
*   [x] Implement **Trigger Manager** (Webhook listener + Scheduler).
*   [x] Integrate **Face Recognition**.
*   [x] Integrate **AI Provider**.
*   [x] Implement **Camera-Specific Prompts** logic.

### Phase 3: Persistence & Polish
*   [x] Database & Timeline UI.
*   [x] Face Management.
*   [ ] Analytics Dashboard.
*   [ ] WhatsApp Notification Integration.

## 6. Directory Structure
```
velovision/
├── docker-compose.yml
├── Dockerfile
├── src/
│   ├── main.py            # FastAPI entry point
│   ├── streaming.py       # RTSP -> MJPEG
│   ├── analysis.py        # AI Logic
│   ├── triggers.py        # Scheduler & Webhook handlers
│   ├── database.py        # Models
│   ├── routers/
│   │   ├── api.py         # API Endpoints
│   │   └── ui.py          # UI Routes
│   └── templates/
└── data/ (Mounted)
    ├── faces/
    ├── events/
    └── config.yaml
```

## 7. Configuration Strategy (config.yaml)
```yaml
cameras:
  front_door:
    rtsp_url: "rtsp://..."
    analysis_prompt: "You are a doorbell cam. Check for visitors."
    message_prompt: "Summarize the event in 1 sentence for a notification."
    triggers:
      - type: webhook
        id: "front_motion"
      - type: schedule
        every_hours: 1
        every_minutes: 0
```
