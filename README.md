# 🤖 Aura — AI Companion Device

> A real-time, vision-driven AI companion that observes, understands, and responds to the user's presence, emotion, gestures, and environment.

---

## Repository Structure

```
aura-companion/
├── main.py                          ← Full pipeline entry point
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
│
├── config/
│   ├── config.py                   ← Pydantic settings loader
│   └── settings.yaml               ← Master configuration
│
├── perception/                     ← Vision perception pipeline
│   ├── orchestrator.py             ← Tiered async pipeline coordinator
│   ├── camera/
│   │   ├── capture.py              ← Thread-safe camera acquisition
│   │   └── processor.py            ← Frame pre-processing (CLAHE, resize)
│   ├── detection/
│   │   └── human_detector.py       ← YOLOv8 human presence detection
│   ├── recognition/
│   │   └── face_recognizer.py      ← InsightFace/ArcFace identity system
│   ├── emotion/
│   │   └── emotion_detector.py     ← DeepFace emotion recognition
│   ├── gesture/
│   │   └── gesture_recognizer.py   ← MediaPipe hands + pose
│   ├── objects/
│   │   └── object_detector.py      ← YOLOv8 contextual objects
│   └── scene/
│       └── vlm_analyzer.py         ← Qwen-VL / LLaVA scene understanding
│
├── memory/
│   ├── temporal_buffer.py          ← Rolling 30s perception buffer
│   └── session_memory.py           ← Per-session activity/emotion log
│
├── context/
│   └── context_engine.py           ← Perception → ContextObject synthesis
│
├── personalization/
│   ├── user_profile.py             ← User profile data + JSON storage
│   └── personalization_engine.py   ← Preference learning + behavior instructions
│
├── agent/
│   └── aura_agent.py               ← LLM agent (Ollama/Claude/OpenAI)
│
├── behavior/
│   ├── executor.py                 ← Action execution coordinator
│   ├── tts/
│   │   └── tts_engine.py           ← pyttsx3 / Coqui TTS
│   ├── leds/
│   │   └── led_controller.py       ← NeoPixel LED ring control
│   └── animations/                 ← Animation definitions (future)
│
├── hardware/
│   └── microcontroller.py          ← ESP32/Arduino serial bridge
│
├── scripts/
│   ├── enroll.py                   ← Face enrollment CLI
│   ├── list_users.py               ← Show enrolled users
│   └── test_camera.py              ← Camera diagnostic
│
├── tests/
│   ├── unit/
│   │   ├── test_temporal_buffer.py
│   │   ├── test_context_engine.py
│   │   └── test_agent.py
│   └── integration/
│
└── data/
    ├── face_db/                    ← Enrolled face embeddings (gitignored)
    ├── profiles/                   ← User JSON profiles (gitignored)
    ├── sessions/                   ← Session SQLite DB (gitignored)
    └── logs/                       ← Rotating log files (gitignored)
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env to set ANTHROPIC_API_KEY or confirm ollama is running
```

Optionally edit `config/settings.yaml` to change camera device, model paths, or agent backend.

### 3. Enroll a user

```bash
python scripts/enroll.py
```

Follow the prompts to capture enrollment frames.

### 4. Run Aura

```bash
# Standard run (requires Ollama running locally)
python main.py

# With live video debug window
python main.py --show-video

# Disable VLM for faster startup on resource-constrained hardware
python main.py --no-vlm

# Dry run (perception only, no TTS/LEDs)
python main.py --dry-run
```

---

## Configuration

All configuration lives in `config/settings.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `camera` | Device ID, resolution, FPS |
| `detection` | YOLOv8 model variant, confidence threshold |
| `recognition` | InsightFace model, similarity threshold |
| `scene` | VLM model name, run interval |
| `agent` | LLM backend (ollama/anthropic/openai), model |
| `cooldowns` | Per-action repeat prevention intervals |
| `hardware` | Serial port for microcontroller, LED count |

Environment variables in `.env` override `settings.yaml` values.

---

## Agent Backends

| Backend | Config Value | Setup Required |
|---------|-------------|----------------|
| **Ollama (local)** | `ollama` | `ollama pull llama3.1:8b` |
| **Anthropic Claude** | `anthropic` | Set `ANTHROPIC_API_KEY` in `.env` |
| **OpenAI** | `openai` | Set `OPENAI_API_KEY` in `.env` |

---

## Running Tests

```bash
pytest                        # All tests
pytest tests/unit/            # Unit tests only
pytest -k "test_agent"        # Specific test module
pytest --cov=. --cov-report=html  # With coverage report
```

---

## Hardware Setup (Optional)

For full physical companion device operation:

1. Flash `firmware/esp32_companion.ino` to an ESP32 (or Arduino Nano 33)
2. Set `hardware.serial_enabled: true` in `settings.yaml`
3. Set `hardware.serial_port` to the correct COM/tty port
4. Connect NeoPixel ring to pin 6 and servos to pins 9/10

Without hardware, all LED and servo commands are simulated and logged to console.

---

## Privacy

All processing occurs on-device by default. No video, audio, or biometric data is transmitted unless you configure a cloud LLM backend (`anthropic` or `openai`). Face embeddings and user profiles are stored locally in `data/`.

---

## License

MIT — see LICENSE for details.
