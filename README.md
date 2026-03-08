# 🤖 Aura — AI Companion Device

> A real-time, vision-driven AI companion that observes, understands, and responds to the user's presence, emotion, gestures, and environment. Built with a modular, tiered perception pipeline and multi-backend LLM support.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [System Components](#system-components)
- [AI Models](#ai-models)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Hardware Integration](#hardware-integration)
- [Security & Privacy](#security--privacy)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

Aura is an intelligent companion device that uses computer vision and natural language processing to:

- **See** — Detect humans, recognize faces, analyze emotions, and understand gestures
- **Understand** — Build contextual awareness of the user's activity, mood, and environment
- **Remember** — Maintain short-term and session-based memory of interactions
- **Personalize** — Learn user preferences and adapt behavior over time
- **Respond** — Generate contextual speech, LED animations, and physical movements

### Key Features

- 🎯 **Multi-Tier Perception Pipeline** — Optimized real-time processing (30 FPS Tier 1, async Tier 2/3)
- 👤 **Face Recognition** — Enroll and recognize multiple users with confidence scoring
- 😊 **Emotion Detection** — Real-time analysis of 7 emotional states with temporal smoothing
- 👋 **Gesture Recognition** — Wave detection, hand gestures, and body orientation tracking
- 🧠 **Contextual AI Agent** — LLM-powered decision making with cooldown management
- 🔊 **Text-to-Speech** — Multiple TTS backends with tone-adaptive speech rates
- 💡 **NeoPixel LED Control** — Color-coded emotional responses and animations
- 🔧 **Modular Hardware Support** — ESP32/Arduino integration for servos and displays

---

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AURA AI COMPANION                              │
│                         ┌─────────────────────┐                             │
│                         │   PERCEPTION LAYER  │                             │
│                         │   (Vision Pipeline) │                             │
└─────────────────────────┴─────────────────────┴─────────────────────────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
       ▼                              ▼                              ▼
┌──────────────┐           ┌──────────────────┐           ┌──────────────────┐
│   TIER 1     │           │     TIER 2       │           │     TIER 3       │
│ (Real-time)  │           │  (Every ~3s)     │           │ (Every ~10s)     │
│  30 FPS      │           │  Async Thread    │           │  Async Thread    │
├──────────────┤           ├──────────────────┤           ├──────────────────┤
│ • Human      │           │ • Face           │           │ • VLM Scene      │
│   Detection  │           │   Recognition    │           │   Analysis       │
│   (YOLOv8)   │           │   (InsightFace)  │           │   (Qwen-VL/      │
│ • Gesture    │           │ • Emotion        │           │    LLaVA)        │
│   Tracking   │           │   Detection      │           │                  │
│   (MediaPipe)│           │   (DeepFace)     │           │ • Context Build  │
└──────────────┘           │ • Object         │           │ • Agent Decision │
       │                   │   Detection      │           └──────────────────┘
       │                   │   (YOLOv8)       │                   │
       │                   └──────────────────┘                   │
       │                            │                             │
       └────────────────────────────┼─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │      MEMORY SYSTEM          │
                    ├─────────────────────────────┤
                    │ • Temporal Buffer (30s)     │
                    │ • Session Memory            │
                    │ • User Profiles (JSON)      │
                    │ • Episodic Memory (Chroma)  │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │     CONTEXT ENGINE          │
                    ├─────────────────────────────┤
                    │ Synthesizes perception      │
                    │ outputs + memory into       │
                    │ structured ContextObject    │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   PERSONALIZATION ENGINE    │
                    ├─────────────────────────────┤
                    │ • User preference learning  │
                    │ • Behavior instruction      │
                    │   generation                │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │        AI AGENT             │
                    ├─────────────────────────────┤
                    │ Multi-backend LLM support:  │
                    │ • Ollama (local)            │
                    │ • Anthropic Claude          │
                    │ • OpenAI GPT                │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │    BEHAVIOR EXECUTOR      │
                    ├─────────────────────────────┤
                    │ • TTS (pyttsx3/Coqui)       │
                    │ • LED Controller            │
                    │ • Animation Player          │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │    HARDWARE INTERFACE       │
                    ├─────────────────────────────┤
                    │ • Serial Bridge (ESP32)   │
                    │ • NeoPixel LED Ring         │
                    │ • Servo Motors              │
                    │ • OLED Display              │
                    └─────────────────────────────┘
```

### Data Flow

```
Camera Frame → Frame Processor → PerceptionOrchestrator
                                           │
       ┌───────────────────────────────────┼───────────────────────────────────┐
       │                                   │                                   │
       ▼                                   ▼                                   ▼
┌──────────────┐              ┌──────────────────────┐            ┌─────────────────┐
│  Tier 1      │              │      Tier 2          │            │     Tier 3      │
│ Synchronous  │              │  Async ThreadPool    │            │  Async Thread   │
├──────────────┤              ├──────────────────────┤            ├─────────────────┤
│ • Human      │              │ • Face Recognition   │            │ • VLM Scene     │
│   Detection  │              │ • Emotion Analysis   │            │   Understanding │
│ • Gesture    │              │ • Object Detection   │            │                 │
│   Recognition│              └──────────────────────┘            └─────────────────┘
└──────────────┘                          │                                   │
       │                                  │                                   │
       └──────────────────────────────────┼───────────────────────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │  Temporal Memory     │
                              │  Buffer (30s window) │
                              └──────────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │   Context Engine     │
                              │  (build_context)     │
                              └──────────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │  Personalization     │
                              │  Engine + Profile    │
                              └──────────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │    Aura Agent        │
                              │   (LLM decision)     │
                              └──────────────────────┘
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │  Behavior Executor   │
                              │ • TTS • LEDs • Anim  │
                              └──────────────────────┘
```

---

## System Components

### 1. Perception Layer (`/perception`)

| Component | Technology | Frequency | Purpose |
|-----------|------------|-----------|---------|
| **Camera** | OpenCV (V4L2/DShow) | 30 FPS | Frame acquisition & preprocessing |
| **Human Detection** | YOLOv8 (Nano/Small) | Every frame | Person detection & presence gating |
| **Face Recognition** | InsightFace (ArcFace) | Every 3s | Identity verification & enrollment |
| **Emotion Detection** | DeepFace (FER+) | Every 3s | 7-class emotion analysis with smoothing |
| **Gesture Recognition** | MediaPipe Hands+Pose | Every frame | Hand gestures, waving, body orientation |
| **Object Detection** | YOLOv8 | Every 3s | Contextual object awareness |
| **Scene Analysis** | Qwen-VL / LLaVA | Every 10s | High-level scene understanding (optional) |

### 2. Memory System (`/memory`)

| Component | Storage | Purpose |
|-----------|---------|---------|
| **Temporal Buffer** | In-memory (deque) | Rolling 30-second perception snapshots |
| **Session Memory** | In-memory + SQLite | Per-session activity, emotion, interaction logs |
| **User Profiles** | JSON files | Persistent user preferences & behavior patterns |
| **Episodic Memory** | ChromaDB | Long-term vector memory (future extension) |

### 3. Context Engine (`/context`)

Synthesizes all perception outputs into a structured `ContextObject` containing:
- User identity & confidence
- Emotional state & trends
- Activity classification
- Gesture & body orientation
- Detected objects & scene description
- Temporal context (time of day, session duration)
- Event classification (new user, waving, extended session, etc.)

### 4. Personalization Engine (`/personalization`)

- **User Profiles**: JSON-based storage of preferences
- **Preference Learning**: Exponential moving average updates from feedback
- **Behavior Instructions**: Dynamic prompt generation for the LLM agent

### 5. AI Agent (`/agent`)

Multi-backend LLM agent supporting:
- **Ollama** (default) — Local inference with llama3.1:8b
- **Anthropic Claude** — Cloud API with claude-haiku-4-5-20251001
- **OpenAI GPT** — Cloud API with gpt-4o-mini

Features:
- JSON-structured responses
- Action cooldown management
- Tone-aware response generation
- Conversation history tracking

### 6. Behavior Execution (`/behavior`)

| Component | Backends | Features |
|-----------|----------|----------|
| **TTS Engine** | pyttsx3, Coqui TTS | Tone-adaptive speech rates, async playback |
| **LED Controller** | NeoPixel via Serial | Color-coded emotions, pulse animations |
| **Executor** | — | Urgency-based scheduling, action coordination |

### 7. Hardware Interface (`/hardware`)

- **Microcontroller Bridge**: Serial JSON protocol to ESP32/Arduino
- **Servo Control**: Head movement (yaw/pitch)
- **Display Output**: Text/emoji display on OLED

---

## AI Models

### Computer Vision Models

| Model | Provider | Purpose | Size | Device |
|-------|----------|---------|------|--------|
| **YOLOv8n/s** | Ultralytics | Human/Object detection | ~6MB / 22MB | CPU/CUDA |
| **buffalo_sc/l** | InsightFace | Face detection & recognition | ~100MB | CPU/GPU |
| **DeepFace** | serengil | Emotion recognition | ~100MB | CPU |
| **MediaPipe** | Google | Hand & pose landmarks | Built-in | CPU |

### Language Models

| Backend | Model | Context | Best For |
|---------|-------|---------|----------|
| **Ollama** | llama3.1:8b | 128K | Privacy-first, offline operation |
| **Anthropic** | claude-haiku-4-5-20251001 | 200K | Speed & quality balance |
| **OpenAI** | gpt-4o-mini | 128K | High-quality responses |

### Vision-Language Models (Optional)

| Model | Provider | Purpose |
|-------|----------|---------|
| **Qwen-VL-Chat** | Alibaba | Scene understanding |
| **LLaVA-1.5** | Haotian Liu | Visual question answering |

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- Ollama (for local LLM)
- USB Camera (UVC compatible)
- (Optional) ESP32 or Arduino for hardware features

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/aura-companion.git
cd aura-companion
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download AI Models

The system will auto-download required models on first run, or you can pre-download:

```bash
# YOLOv8 models
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8s.pt

# InsightFace models (auto-downloaded on first use)
# DeepFace models (auto-downloaded on first use)
```

### Step 5: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# LLM Configuration
AURA_AGENT_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Optional: Cloud LLM APIs
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Camera
CAMERA_DEVICE_ID=0

# Hardware (optional)
SERIAL_ENABLED=false
SERIAL_PORT=/dev/ttyUSB0  # Linux/Mac
# SERIAL_PORT=COM3        # Windows
```

### Step 6: Setup Ollama (for local LLM)

```bash
# Install Ollama from https://ollama.com

# Pull the default model
ollama pull llama3.1:8b
```

### Step 7: Enroll First User

```bash
python scripts/enroll.py
```

Follow prompts to capture face samples for recognition.

---

## Configuration

### Main Configuration File

All settings are in `config/settings.yaml`:

```yaml
# ── Camera ──────────────────────────────────────
camera:
  device_id: 0              # Camera index (0 for default)
  width: 1280               # Capture resolution
  height: 720
  fps: 30                   # Target FPS
  backend: "dshow"          # "v4l2" (Linux), "dshow" (Windows), "avfoundation" (Mac)

# ── Detection (YOLOv8) ────────────────────────
detection:
  model_path: "yolov8n.pt"  # Model variant (n/s/m/l)
  confidence_threshold: 0.45
  device: "cpu"             # "cpu", "cuda", "mps"

# ── Face Recognition ──────────────────────────
recognition:
  model_name: "buffalo_sc"  # "buffalo_sc" (fast) or "buffalo_l" (accurate)
  recognition_threshold: 0.45
  database_path: "data/face_db/face_db.pkl"

# ── Emotion Detection ─────────────────────────
emotion:
  backend: "opencv"         # Face detector backend
  smoothing_window_frames: 10

# ── Gesture Recognition ──────────────────────
gesture:
  max_num_hands: 2
  min_detection_confidence: 0.6
  pose_model_complexity: 1  # 0=lite, 1=full, 2=heavy

# ── AI Agent ──────────────────────────────────
agent:
  backend: "ollama"         # "ollama", "anthropic", "openai"
  ollama_model: "llama3.1:8b"
  anthropic_model: "claude-haiku-4-5-20251001"
  openai_model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 512

# ── TTS ───────────────────────────────────────
behavior:
  tts_engine: "pyttsx3"     # "pyttsx3" or "coqui"
  tts_rate: 175             # Speech rate (words/min)
  tts_volume: 0.9

# ── Hardware ──────────────────────────────────
hardware:
  serial_enabled: false
  serial_port: "/dev/ttyUSB0"
  serial_baud: 115200
  led_count: 24
  servo_enabled: false
```

### Environment Variables

Environment variables override YAML settings:

| Variable | Description |
|----------|-------------|
| `AURA_AGENT_BACKEND` | LLM backend (ollama/anthropic/openai) |
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `OLLAMA_BASE_URL` | Ollama server URL |
| `OLLAMA_MODEL` | Ollama model name |
| `CAMERA_DEVICE_ID` | Camera device index |
| `AURA_DEBUG` | Enable debug mode (true/false) |
| `AURA_LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `SERIAL_PORT` | Serial port path |
| `SERIAL_ENABLED` | Enable serial hardware (true/false) |

---

## Usage

### Running Aura

```bash
# Standard run
python main.py

# With live video debug window
python main.py --show-video

# Disable VLM (faster startup)
python main.py --no-vlm

# Dry run (perception only, no TTS/LEDs)
python main.py --dry-run

# Custom config file
python main.py --config my_settings.yaml

# Run enrollment wizard
python main.py --enroll
```

### User Enrollment

```bash
# Interactive enrollment
python scripts/enroll.py

# Command-line enrollment
python scripts/enroll.py --user-id user_001 --name "Alex" --frames 15

# List enrolled users
python scripts/list_users.py

# Test camera
python scripts/test_camera.py
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--config PATH` | Path to YAML config file |
| `--enroll` | Run enrollment wizard |
| `--no-vlm` | Disable VLM scene analysis |
| `--show-video` | Show debug video window |
| `--dry-run` | Skip TTS and LED output |

---

## Hardware Integration

### Supported Hardware

- **Microcontroller**: ESP32, Arduino Nano 33, or compatible
- **LED Ring**: WS2812B NeoPixel (24 LEDs recommended)
- **Servos**: 2x SG90 or MG90S for head movement
- **Display**: SSD1306 128x64 OLED (optional)

### Wiring Diagram

```
ESP32/Arduino
│
├── GPIO 6  ──► NeoPixel Data In (via 470Ω resistor)
├── GPIO 9  ──► Servo 1 (Yaw - horizontal)
├── GPIO 10 ──► Servo 2 (Pitch - vertical)
├── TX/RX   ──► USB Serial to PC
└── 5V/GND  ──► Power (separate supply recommended for LEDs)
```

### Firmware

Flash the Arduino/ESP32 firmware located in `firmware/esp32_companion.ino` (if available) or create your own using the JSON protocol:

```json
{"cmd": "led_color", "r": 255, "g": 100, "b": 50}
{"cmd": "servo_move", "yaw": 90, "pitch": 75}
{"cmd": "display_text", "text": "Hello!", "color": "white"}
```

### Enabling Hardware

Edit `config/settings.yaml`:

```yaml
hardware:
  serial_enabled: true
  serial_port: "/dev/ttyUSB0"  # Linux/Mac
  # serial_port: "COM3"        # Windows
  serial_baud: 115200
  led_count: 24
  servo_enabled: true
```

---

## Security & Privacy

### Data Storage

| Data Type | Location | Encrypted |
|-----------|----------|-----------|
| Face embeddings | `data/face_db/face_db.pkl` | No |
| User profiles | `data/profiles/*.json` | No |
| Session logs | `data/sessions/sessions.db` | No |
| Logs | `data/logs/*.log` | No |

### Privacy Features

- ✅ **On-device processing** — All vision AI runs locally
- ✅ **No cloud video transmission** — Camera data never leaves the device
- ✅ **Local LLM option** — Use Ollama for fully offline operation
- ✅ **Configurable cloud LLM** — Only text context sent when using Anthropic/OpenAI

### Security Recommendations

1. **Physical Access Control**
   - Keep the device in physically secure locations
   - Prevent unauthorized access to stored face embeddings

2. **API Key Management**
   - Store API keys in `.env` file (never commit to git)
   - Use environment-specific keys
   - Rotate keys periodically

3. **Face Database Protection**
   ```bash
   # Restrict access to face database (Linux/Mac)
   chmod 600 data/face_db/face_db.pkl
   ```

4. **Network Security**
   - If using Ollama remotely, ensure HTTPS/TLS
   - Firewall the Ollama port (11434) from external access

5. **Audit Logging**
   - Enable debug logging to track access: `AURA_DEBUG=true`
   - Review logs regularly: `tail -f data/logs/aura.log`

### Data Retention

- Face embeddings: Persistent until manually deleted
- Session logs: Persistent (SQLite with rotation)
- User profiles: Persistent until manually deleted
- Logs: Rotated daily with configurable retention

---

## Customization

### Creating Custom Agent Behaviors

Edit `config/settings.yaml` cooldowns:

```yaml
cooldowns:
  greet: 300        # 5 minutes between greetings
  reminder: 600   # 10 minutes between reminders
  comment: 180    # 3 minutes between comments
  question: 300   # 5 minutes between questions
  reaction: 30    # 30 seconds between reactions
```

### Custom LED Animations

Modify `behavior/leds/led_controller.py`:

```python
def my_custom_animation(self, color):
    """Add your custom LED sequence."""
    # Implementation here
    pass
```

### Custom Prompts

Edit the prompt template in `agent/aura_agent.py`:

```python
_PROMPT = """You are Aura, a helpful AI companion. ..."""
```

### Adding New Perception Modules

1. Create module in `perception/<module>/`
2. Implement `initialize()` and `process()` methods
3. Register in `perception/orchestrator.py`
4. Add configuration in `config/config.py`

---

## Troubleshooting

### Common Issues

#### Camera Not Found

```bash
# List available cameras
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"

# Test camera
python scripts/test_camera.py
```

#### CUDA/GPU Issues

```yaml
# In settings.yaml, force CPU mode:
detection:
  device: "cpu"
recognition:
  ctx_id: -1  # -1=CPU, 0=GPU
```

#### Ollama Connection Failed

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Check model is downloaded
ollama list
```

#### Face Recognition Not Working

1. Ensure user is enrolled: `python scripts/enroll.py`
2. Check database exists: `ls data/face_db/`
3. Verify recognition threshold: Adjust `recognition_threshold` in config

#### Import Errors (MediaPipe)

```bash
# If MediaPipe 0.10+ causes issues
pip install mediapipe==0.9.3.0
```

### Performance Optimization

| Issue | Solution |
|-------|----------|
| Low FPS | Disable VLM: `--no-vlm` or `scene.enabled: false` |
| High CPU | Use YOLOv8n (nano) instead of YOLOv8s |
| Slow responses | Switch to faster LLM (claude-haiku) |
| Memory usage | Reduce `short_term_window_seconds` |

### Debug Mode

```bash
# Enable verbose logging
export AURA_DEBUG=true
export AURA_LOG_LEVEL=DEBUG
python main.py
```

---

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=. --cov-report=html
```

### Project Structure

```
├── main.py                  # Application entry point
├── config/                # Configuration management
│   ├── config.py         # Pydantic settings
│   └── settings.yaml     # Default configuration
├── perception/            # Vision pipeline
│   ├── orchestrator.py   # Tiered processing coordinator
│   ├── camera/           # Capture & preprocessing
│   ├── detection/        # Human detection
│   ├── recognition/      # Face recognition
│   ├── emotion/          # Emotion detection
│   ├── gesture/          # Gesture recognition
│   ├── objects/          # Object detection
│   └── scene/            # VLM scene analysis
├── memory/               # Memory systems
│   ├── temporal_buffer.py
│   └── session_memory.py
├── context/              # Context synthesis
│   └── context_engine.py
├── personalization/      # User profiles & learning
│   ├── user_profile.py
│   └── personalization_engine.py
├── agent/                # LLM agent
│   └── aura_agent.py
├── behavior/             # Output execution
│   ├── executor.py
│   ├── tts/
│   └── leds/
├── hardware/             # Serial communication
│   └── microcontroller.py
├── scripts/              # CLI utilities
├── tests/                # Test suite
└── data/                 # Runtime data (gitignored)
    ├── face_db/
    ├── profiles/
    ├── sessions/
    └── logs/
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [InsightFace](https://github.com/deepinsight/insightface)
- [DeepFace](https://github.com/serengil/deepface)
- [MediaPipe](https://mediapipe.dev/)
- [Ollama](https://ollama.com/)

---

**Built with ❤️ for AI companions everywhere**