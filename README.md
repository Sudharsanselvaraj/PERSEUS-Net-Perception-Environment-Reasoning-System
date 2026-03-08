# PERSEUS-Net
### Perception · Environment · Reasoning · Understanding · System

> A real-time, modular AI perception system that fuses computer vision, temporal memory, and LLM-based reasoning to understand human presence, activity, emotion, and context — built for edge-deployable companion intelligence.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-purple?style=flat-square)](https://ultralytics.com)
[![InsightFace](https://img.shields.io/badge/Recognition-InsightFace-orange?style=flat-square)](https://insightface.ai)
[![MediaPipe](https://img.shields.io/badge/Gesture-MediaPipe-green?style=flat-square)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## Table of Contents

- [What is PERSEUS-Net](#what-is-perseus-net)
- [System Architecture](#system-architecture)
- [Pipeline Tiers](#pipeline-tiers)
- [Module Reference](#module-reference)
- [AI Models](#ai-models)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Hardware Integration](#hardware-integration)
- [Security and Privacy](#security-and-privacy)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## What is PERSEUS-Net

PERSEUS-Net is an edge-deployable AI perception and reasoning system designed for human-aware companion intelligence. It continuously observes an environment through a camera, builds a structured understanding of the human user, and drives intelligent, context-aware responses through a multi-backend LLM agent.

The system name reflects its core capability stack:

| Letter | Meaning | System Layer |
|--------|---------|-------------|
| **P** | Perception | Vision pipeline — detection, recognition, emotion, gesture |
| **E** | Environment | Scene understanding — objects, lighting, spatial context |
| **R** | Reasoning | LLM agent — decision making, action selection |
| **S** | Understanding | Context engine — synthesizes all signals into meaning |
| **E** | Episodic | Memory system — temporal buffer, session log, profiles |
| **U** | User-aware | Personalization — preference learning, behavior adaptation |
| **S** | System | Hardware abstraction — TTS, LEDs, serial bridge |

### Core Capabilities

- **Tiered Real-Time Vision Pipeline** — Tier 1 at 30 FPS (detection/gesture), Tier 2 async every 3s (face/emotion/objects), Tier 3 async every 10s (VLM scene analysis)
- **Multi-User Face Recognition** — ArcFace embeddings with cosine similarity matching and on-disk enrollment database
- **Temporal Emotion Tracking** — 7-class emotion detection with rolling smoothing buffer and valence/trend analysis
- **Gesture Intelligence** — MediaPipe-based hand gesture classification + waving detection via oscillation tracking
- **Contextual Object Awareness** — COCO-class object detection with activity inference rules
- **LLM Agent (Aura)** — Multi-backend reasoning agent with structured JSON output, cooldown enforcement, and tone-adaptive responses
- **Personalization Engine** — Preference learning via exponential moving average updates from interaction feedback
- **Hardware Output Layer** — TTS, NeoPixel LED ring, ESP32/Arduino serial bridge for servo and display control

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         PERSEUS-Net SYSTEM                               ║
╚══════════════════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────────────────┐
  │                        HARDWARE LAYER                                │
  │   USB/CSI Camera  →  Edge Compute (CPU/GPU)  →  Output Peripherals   │
  └────────────────────────────────┬─────────────────────────────────────┘
                                   │ Raw BGR frames @ 30 FPS
  ┌────────────────────────────────▼─────────────────────────────────────┐
  │                      FRAME ACQUISITION                               │
  │   CameraInputLayer (threaded, bounded queue)                         │
  │   FrameProcessor   (CLAHE, resize, quality gating)                   │
  └────────────────────────────────┬─────────────────────────────────────┘
                                   │
          ┌────────────────────────┼─────────────────────────────┐
          │                        │                             │
          ▼  TIER 1 (sync/frame)   ▼  TIER 2 (async/3s)         ▼  TIER 3 (async/10s)
  ┌───────────────┐       ┌────────────────────┐       ┌──────────────────────┐
  │ YOLOv8-nano   │       │ InsightFace ArcFace│       │ Qwen-VL / LLaVA      │
  │ Human Detect  │       │ Face Recognition   │       │ VLM Scene Analysis   │
  │               │       ├────────────────────┤       └──────────────────────┘
  │ MediaPipe     │       │ DeepFace CNN       │
  │ Gesture+Pose  │       │ Emotion Detection  │
  └───────┬───────┘       ├────────────────────┤
          │               │ YOLOv8-small       │
          │               │ Object Detection   │
          │               └────────┬───────────┘
          │                        │
          └────────────────────────┼──────────────────────────────┐
                                   │                              │
  ┌────────────────────────────────▼─────────────────────────────▼──────┐
  │                     PERCEPTION ORCHESTRATOR                         │
  │   ThreadPoolExecutor — schedules tiers, manages shared state        │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │ PerceptionState
  ┌────────────────────────────────▼────────────────────────────────────┐
  │                      MEMORY SYSTEM                                  │
  │   TemporalMemoryBuffer  — rolling 30s snapshots, trend analysis     │
  │   SessionMemory         — per-session activity/emotion/interaction  │
  │   ProfileStore          — persistent JSON user profiles             │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │ BufferSummary + SessionMemory
  ┌────────────────────────────────▼────────────────────────────────────┐
  │                      CONTEXT ENGINE                                 │
  │   Synthesizes perception + memory → structured ContextObject        │
  │   Fields: user_id, emotion, emotion_trend, activity, gesture,       │
  │           detected_objects, scene_description, event, time_of_day   │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │ ContextObject
  ┌────────────────────────────────▼────────────────────────────────────┐
  │                  PERSONALIZATION ENGINE                             │
  │   UserProfile lookup → behavior instructions for LLM prompt         │
  │   Preference learning via EMA on interaction outcome feedback       │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │ ContextObject + BehaviorInstructions
  ┌────────────────────────────────▼────────────────────────────────────┐
  │                        AURA AGENT                                   │
  │   LLM backend: Ollama | Anthropic Claude | OpenAI GPT               │
  │   Structured JSON output → AgentAction                              │
  │   Cooldown enforcement per action type                              │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │ AgentAction
  ┌────────────────────────────────▼────────────────────────────────────┐
  │                    BEHAVIOR EXECUTOR                                │
  │   TTS (pyttsx3 / Coqui)  →  Speech output                           │
  │   LEDController           →  NeoPixel ring color/animation          │
  │   MicrocontrollerBridge   →  Serial JSON to ESP32/Arduino           │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Tiers

PERSEUS-Net uses a three-tier asynchronous processing architecture to balance real-time responsiveness with computational cost:

### Tier 1 — Synchronous (Every Frame, ~30 FPS)

| Module | Model | Latency Target |
|--------|-------|---------------|
| Human Detection | YOLOv8-nano | < 20ms CPU / < 5ms GPU |
| Gesture + Pose | MediaPipe Hands + Pose | < 10ms CPU |
| Frame Quality Gate | OpenCV Laplacian/CLAHE | < 2ms |

Tier 1 runs in the **main loop thread**. All downstream tiers are skipped if no human is detected.

### Tier 2 — Asynchronous (Every ~3 Seconds)

| Module | Model | Latency Target |
|--------|-------|---------------|
| Face Recognition | InsightFace buffalo_sc / buffalo_l | < 200ms CPU |
| Emotion Detection | DeepFace (FER+/AffectNet backend) | < 500ms CPU |
| Object Detection | YOLOv8-small | < 100ms CPU |

Tier 2 runs in a **ThreadPoolExecutor worker**. Results are written to shared state protected by a threading lock and read by the main loop asynchronously.

### Tier 3 — Asynchronous (Every ~10 Seconds / Event-Driven)

| Module | Model | Latency Target |
|--------|-------|---------------|
| VLM Scene Analysis | Qwen-VL-Chat / LLaVA-1.5 / MiniCPM-V | 500ms – 3s |
| Context Engine Build | Pure Python synthesis | < 10ms |
| Agent LLM Decision | Ollama / Claude / GPT | 200ms – 2s |

Tier 3 triggers a **full context build and agent decision** via callback (`on_tier3_complete`). This is where Aura decides whether and how to respond.

---

## Module Reference

### Repository Structure

```
PERSEUS-Net/
│
├── main.py                              ← System entry point, full pipeline wiring
│
├── config/
│   ├── config.py                        ← Pydantic v2 typed settings loader
│   └── settings.yaml                    ← Master configuration (all defaults)
│
├── perception/
│   ├── orchestrator.py                  ← Tiered async pipeline coordinator
│   ├── camera/
│   │   ├── capture.py                   ← Thread-safe camera acquisition (bounded queue)
│   │   └── processor.py                 ← CLAHE enhancement, resize, quality gating
│   ├── detection/
│   │   └── human_detector.py            ← YOLOv8 person detection + spatial metadata
│   ├── recognition/
│   │   └── face_recognizer.py           ← InsightFace ArcFace + cosine similarity DB
│   ├── emotion/
│   │   └── emotion_detector.py          ← DeepFace emotion + temporal smoothing
│   ├── gesture/
│   │   └── gesture_recognizer.py        ← MediaPipe Hands + Pose + waving detection
│   ├── objects/
│   │   └── object_detector.py           ← COCO-class detection + activity inference
│   └── scene/
│       └── vlm_analyzer.py              ← VLM scene understanding (Qwen-VL/LLaVA)
│
├── memory/
│   ├── temporal_buffer.py               ← Rolling 30s PerceptionSnapshot buffer
│   └── session_memory.py                ← Per-session activity/emotion/interaction log
│
├── context/
│   └── context_engine.py                ← Perception → ContextObject synthesis
│
├── personalization/
│   ├── user_profile.py                  ← UserProfile dataclass + JSON persistence
│   └── personalization_engine.py        ← EMA preference learning + prompt generation
│
├── agent/
│   └── aura_agent.py                    ← LLM agent (Ollama/Anthropic/OpenAI backends)
│
├── behavior/
│   ├── executor.py                      ← Action execution coordinator
│   ├── tts/
│   │   └── tts_engine.py                ← pyttsx3 + Coqui TTS with tone-adaptive rate
│   └── leds/
│       └── led_controller.py            ← NeoPixel LED ring control + pulse animations
│
├── hardware/
│   └── microcontroller.py               ← JSON serial protocol bridge to ESP32/Arduino
│
├── scripts/
│   ├── enroll.py                        ← Interactive face enrollment CLI
│   ├── list_users.py                    ← Show enrolled user profiles
│   └── test_camera.py                   ← Camera diagnostic with live quality metrics
│
├── tests/
│   ├── unit/
│   │   ├── test_temporal_buffer.py
│   │   ├── test_context_engine.py
│   │   └── test_agent.py
│   └── integration/
│
└── data/                                ← Runtime data (gitignored)
    ├── face_db/face_db.pkl              ← ArcFace embedding database
    ├── profiles/*.json                  ← Per-user preference profiles
    ├── sessions/sessions.db             ← SQLite session log
    └── logs/                            ← Rotating daily logs
```

---

## AI Models

### Vision Models

| Model | Variant | Purpose | Input | Output |
|-------|---------|---------|-------|--------|
| **YOLOv8** | nano (Tier 1), small (Tier 2) | Human + object detection | BGR frame | BBox, class, confidence |
| **InsightFace** | buffalo_sc / buffalo_l | Face detection + ArcFace embedding | BGR frame | 512-dim embedding |
| **DeepFace** | FER+ / AffectNet | 7-class emotion recognition | Face crop | Emotion scores dict |
| **MediaPipe** | Hands + Pose | 21-point hand landmarks, 33-point body pose | RGB frame | Normalized 3D landmarks |

### Language Models

| Backend | Model | Parameters | Context | Privacy |
|---------|-------|-----------|---------|---------|
| **Ollama** | llama3.1:8b | 8B | 128K | 100% local |
| **Ollama** | llama3.2:3b | 3B | 128K | 100% local (low RAM) |
| **Anthropic** | claude-haiku-4-5-20251001 | — | 200K | Cloud (text only) |
| **OpenAI** | gpt-4o-mini | — | 128K | Cloud (text only) |

### Vision-Language Models (Tier 3, Optional)

| Model | Parameters | Quantization | RAM Required |
|-------|-----------|-------------|-------------|
| Qwen-VL-Chat | 7B | 4-bit (BnB) | ~6 GB |
| LLaVA-1.5-7B | 7B | 4-bit (BnB) | ~6 GB |
| MiniCPM-V-2 | 2B | None | ~4 GB |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip 23.0+
- USB or CSI camera (UVC compatible)
- Ollama (for local LLM) — download from [ollama.com](https://ollama.com)
- CUDA 11.8+ optional for GPU acceleration

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Sudharsanselvaraj/PERSEUS-Net-Perception-Environment-Reasoning-System.git
cd PERSEUS-Net-Perception-Environment-Reasoning-System
```

### Step 2 — Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install Dependencies (Tiered)

Install in tiers to avoid long waits and dependency conflicts:

```bash
# Tier 1 — Core vision (fast, ~2 min)
pip install opencv-python numpy Pillow ultralytics mediapipe
pip install pydantic pydantic-settings python-dotenv PyYAML
pip install loguru rich click pyserial httpx

# Tier 2 — Face and emotion (~5 min)
pip install insightface onnxruntime scikit-learn
pip install deepface tf-keras

# Tier 3 — LLM agent (choose one)
pip install httpx                    # Ollama (no extra install)
pip install anthropic                # Claude API
pip install openai                   # OpenAI API

# Tier 4 — VLM scene analysis (optional, heavy ~10 min)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate sentencepiece
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
AURA_AGENT_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
CAMERA_DEVICE_ID=0
AURA_DEBUG=false
AURA_LOG_LEVEL=INFO
SERIAL_ENABLED=false
```

### Step 5 — Pull Ollama Model

```bash
ollama pull llama3.1:8b        # Standard (~4.7 GB)
# or
ollama pull llama3.2:3b        # Low RAM machines (~2 GB)
```

### Step 6 — Enroll First User

```bash
python scripts/enroll.py
```

---

## Configuration

All configuration is in `config/settings.yaml`. Environment variables in `.env` override YAML values at runtime.

### Camera

```yaml
camera:
  device_id: 0          # Camera index (run test_camera.py to find correct ID)
  width: 1280
  height: 720
  fps: 30
  backend: "dshow"      # dshow (Windows) | v4l2 (Linux) | avfoundation (macOS)
```

### Detection

```yaml
detection:
  model_path: "yolov8n.pt"       # n=fastest, s=balanced, m=accurate
  confidence_threshold: 0.45
  device: "cpu"                  # cpu | cuda | mps
```

### Agent

```yaml
agent:
  backend: "ollama"              # ollama | anthropic | openai
  ollama_model: "llama3.1:8b"
  anthropic_model: "claude-haiku-4-5-20251001"
  openai_model: "gpt-4o-mini"
  max_tokens: 512
  temperature: 0.7
  timeout_seconds: 30.0
```

### Cooldowns (seconds between repeated actions)

```yaml
cooldowns:
  greet: 300        # 5 minutes
  reminder: 600     # 10 minutes
  comment: 180      # 3 minutes
  question: 300     # 5 minutes
  reaction: 30      # 30 seconds
```

### Performance Tuning

```yaml
pipeline:
  tier2_interval_seconds: 3.0    # Increase to reduce CPU load
  tier3_interval_seconds: 10.0
  max_worker_threads: 4

scene:
  enabled: false                 # Disable VLM on low-end hardware
```

---

## Usage

### Running PERSEUS-Net

```bash
# Standard run
python main.py

# With live annotated video window
python main.py --show-video

# Disable VLM for faster startup
python main.py --no-vlm

# Perception only — no TTS or LEDs (development mode)
python main.py --dry-run

# Custom config file
python main.py --config my_config.yaml

# Launch enrollment wizard
python main.py --enroll
```

### Enrollment and User Management

```bash
# Interactive enrollment
python scripts/enroll.py

# Programmatic enrollment
python scripts/enroll.py --user-id user_001 --name "Alex" --frames 12

# View enrolled users and profiles
python scripts/list_users.py

# Camera diagnostic
python scripts/test_camera.py
python scripts/test_camera.py --device 1
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config (default: config/settings.yaml) |
| `--enroll` | Run interactive face enrollment wizard |
| `--no-vlm` | Disable Tier 3 VLM scene analysis |
| `--show-video` | Show live debug video with bounding boxes and labels |
| `--dry-run` | Skip TTS and LED output (perception pipeline only) |

---

## Hardware Integration

### Microcontroller Serial Protocol

PERSEUS-Net communicates with ESP32 or Arduino over USB serial using newline-delimited JSON commands:

```json
{"cmd": "led_color", "r": 255, "g": 100, "b": 50}
{"cmd": "servo_move", "yaw": 90, "pitch": 75}
{"cmd": "play_animation", "name": "wave_back"}
{"cmd": "display_text", "text": "Hello!", "color": "white"}
{"cmd": "play_sound", "file": "chime.wav"}
```

### Wiring Reference

```
ESP32 / Arduino
│
├── GPIO 6   ──[470Ω]──► NeoPixel WS2812B Data In
├── GPIO 9   ──────────► Servo 1 Signal (Yaw)
├── GPIO 10  ──────────► Servo 2 Signal (Pitch)
├── SDA/SCL  ──────────► SSD1306 OLED Display
├── USB      ──────────► PC Serial Bridge
└── 5V / GND ──────────► External power supply
```

### Enabling Hardware

```yaml
hardware:
  serial_enabled: true
  serial_port: "COM3"        # Windows: COM3, COM4 etc.
  # serial_port: "/dev/ttyUSB0"  # Linux
  serial_baud: 115200
  led_count: 24
  servo_enabled: true
```

---

## Security and Privacy

### Data Locations

| Data | Path | Contains |
|------|------|----------|
| Face embeddings | `data/face_db/face_db.pkl` | ArcFace 512-dim vectors (not raw images) |
| User profiles | `data/profiles/*.json` | Preferences, interaction stats |
| Session logs | `data/sessions/sessions.db` | Activity and emotion history |
| System logs | `data/logs/` | Timestamped pipeline logs |

### Privacy Architecture

- All camera processing is **fully on-device** — no video is transmitted
- When using Ollama backend, **no data ever leaves the machine**
- When using Anthropic/OpenAI, only the **text context string** is sent — never images or biometric data
- Face embeddings are mathematical vectors — raw face images are never stored

### API Key Security

```bash
# Keys live only in .env — never committed to git
echo ".env" >> .gitignore
git rm --cached .env 2>/dev/null
```

---

## Troubleshooting

### Camera Not Opening

```bash
# Find available camera devices
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    print(f'Device {i}:', cap.isOpened())
    cap.release()
"
```

Set the correct `device_id` and `backend: "dshow"` (Windows) in `settings.yaml`.

### MediaPipe Protobuf Conflict

```bash
pip uninstall mediapipe protobuf -y
pip install protobuf==4.25.3
pip install mediapipe==0.10.14
```

### Ollama Returns 500 Error

```bash
# Check model is downloaded
ollama list

# Restart Ollama service
taskkill /F /IM ollama.exe    # Windows
ollama serve

# Switch to smaller model if low on RAM
ollama pull llama3.2:3b
# Update settings.yaml: ollama_model: "llama3.2:3b"
```

### Agent Not Speaking (action_type error)

Verify `agent/aura_agent.py` contains `"format": "json"` in the Ollama request:

```bash
# Windows
findstr "format" agent\aura_agent.py

# Should output: "format": "json",
```

### Slow Performance

| Symptom | Fix |
|---------|-----|
| Low FPS | Set `scene.enabled: false`, use `yolov8n.pt` |
| High RAM | Use `llama3.2:3b`, set `use_quantization: true` |
| Slow emotion | Set `emotion.backend: "opencv"` |
| Agent timeout | Increase `agent.timeout_seconds: 30.0` |

### Enable Full Debug Logging

```bash
# Windows PowerShell
$env:AURA_DEBUG="true"
$env:AURA_LOG_LEVEL="DEBUG"
python main.py --show-video
```

---

## Development

### Running Tests

```bash
pytest                              # All tests
pytest tests/unit/                  # Unit tests only
pytest -k "test_agent"              # Specific module
pytest --cov=. --cov-report=html    # With HTML coverage report
```

### Adding a New Perception Module

1. Create `perception/<module>/<module>.py`
2. Implement `initialize()` and a `process(frame)` method returning a typed dataclass
3. Add config fields to `config/config.py` and `config/settings.yaml`
4. Register in `perception/orchestrator.py` under the appropriate tier
5. Include output in `context/context_engine.py` → `ContextObject`

### Switching LLM Backend at Runtime

```bash
# No code changes needed — override via environment variable
$env:AURA_AGENT_BACKEND="anthropic"
$env:ANTHROPIC_API_KEY="sk-ant-..."
python main.py
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object and human detection
- [InsightFace](https://github.com/deepinsight/insightface) — ArcFace face recognition
- [DeepFace](https://github.com/serengil/deepface) — Emotion recognition pipeline
- [MediaPipe](https://mediapipe.dev) — Hand and pose landmark detection
- [Ollama](https://ollama.com) — Local LLM inference runtime
- [Anthropic Claude](https://anthropic.com) — Cloud LLM backend

---

*PERSEUS-Net — Perception · Environment · Reasoning · Understanding · System*
