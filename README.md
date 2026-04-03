# PERSEUS-Net
### Perception · Environment · Reasoning · Understanding · System

> A real-time, modular AI perception and reasoning system that fuses computer vision, temporal memory, behavioral intelligence, and LLM-based reasoning to understand human presence, activity, emotion, and context — built for edge-deployable companion intelligence and fully integrated with the AURA gateway for multi-agent skill execution.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Version](https://img.shields.io/badge/Version-2.0.0-brightgreen?style=flat-square)](.)
[![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-purple?style=flat-square)](https://ultralytics.com)
[![InsightFace](https://img.shields.io/badge/Recognition-InsightFace-orange?style=flat-square)](https://insightface.ai)
[![MediaPipe](https://img.shields.io/badge/Gesture-MediaPipe-green?style=flat-square)](https://mediapipe.dev)
[![AURA](https://img.shields.io/badge/Gateway-AURA_ANP-red?style=flat-square)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## Table of Contents

- [What is PERSEUS-Net](#what-is-perseus-net)
- [What's New in v2.0](#whats-new-in-v20)
- [System Architecture](#system-architecture)
- [Pipeline Architecture](#pipeline-architecture)
- [Module Reference](#module-reference)
- [Intelligence Layer](#intelligence-layer)
- [ANP Gateway Integration](#anp-gateway-integration)
- [Hardware Profiles](#hardware-profiles)
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

PERSEUS-Net is an edge-deployable AI perception and reasoning system designed for human-aware companion intelligence. It continuously observes an environment through a camera, builds a structured understanding of the human user across multiple sensory channels, and drives intelligent context-aware responses through a multi-backend LLM agent — either locally or via the AURA gateway for full multi-agent skill execution.

| Letter | Meaning | System Layer |
|--------|---------|-------------|
| **P** | Perception | Unified vision pipeline — detection, recognition, emotion, gesture, depth, gaze |
| **E** | Environment | Scene understanding — objects, spatial context, audio, VLM analysis |
| **R** | Reasoning | LLM agent + AURA gateway — decision making, skill execution |
| **S** | Understanding | Context engine — synthesizes all signals into structured meaning |
| **E** | Episodic | Memory system — temporal buffer, causal chain, session log, profiles |
| **U** | User-aware | Intelligence layer — behavioral patterns, anomaly detection, predictive triggering |
| **S** | System | Hardware abstraction — TTS, LEDs, serial bridge, ANP node |

### Core Capabilities

- **Unified Adaptive Perception Pipeline** — Single `UnifiedPerceptionWorker` replacing fixed 3-tier architecture. Phase A runs face/objects/gaze/depth in parallel futures. Phase B uses shared face crop for emotion inference. Phase C fires VLM conditionally on elapsed time. Adaptive gating triggers on frame delta, gesture change, new face, or audio utterance — not fixed intervals.
- **Hardware-Agnostic Auto-Profiling** — Detects hardware at startup (Raspberry Pi, Jetson, workstation, cloud). Automatically tunes model sizes, intervals, and which optional modules activate. Single codebase runs everywhere.
- **Multi-User Face Recognition** — ArcFace embeddings with cosine similarity matching, ByteTrack persistent IDs across frames, on-disk enrollment database.
- **EWA Emotion Tracking** — Exponential weighted average replaces simple mean smoothing. Body orientation gating suppresses inference when face is occluded. 7-class detection with valence, arousal, trend analysis.
- **Depth Estimation** — MiDaS monocular depth on every human detection. Replaces bbox-area heuristic proximity with actual centimeter estimates.
- **Gaze Tracking** — L2CS-Net estimates yaw/pitch gaze angle. Determines if user is looking at the device — changes entire interaction trigger logic.
- **Audio Channel** — Silero VAD + faster-whisper STT. Wake-word gating optional. Audio utterances inject directly into perception pipeline triggering immediate context build.
- **Temporal Action Recognition** — MoViNet/SlowFast over rolling frame clips. Classifies activities like "drinking", "typing", "stretching" beyond what single-frame object detection infers.
- **Intelligence Layer** — `CausalChain` tracks cause-effect across interactions. `AnomalyDetector` uses Isolation Forest + rule-based checks for unusual behavioral patterns. `BehaviorModel` learns per-user arrival patterns, emotion timelines, interaction receptivity.
- **ANP Gateway Integration** — PERSEUS-Net registers as an authenticated ANP node to the AURA gateway. Sends structured typed perception events. Receives back action commands (speak, led_pattern, home_assistant_trigger, spotify_play). Full WebSocket with HELLO/WELCOME handshake, auto-reconnect, heartbeat.
- **Local Agent Fallback** — When gateway is offline, falls back to local Aura agent (Ollama / Claude / OpenAI) with full cooldown enforcement.
- **Personalization Engine** — EMA preference learning from interaction feedback. Generates per-user behavioral instructions injected into LLM prompt.
- **Hardware Output Layer** — TTS, NeoPixel LED ring, ESP32/Arduino serial bridge.

---

## What's New in v2.0

### Architecture Changes

| Component | v1.0 | v2.0 |
|-----------|------|------|
| Pipeline | 3 fixed tiers (sync/3s/10s) | Unified worker + adaptive gating |
| Triggering | Fixed intervals only | Frame delta + gesture change + audio injection |
| Emotion smoothing | Simple mean over window | Exponential weighted average (EWA) |
| Emotion gating | None | Suppressed when face occluded by body orientation |
| Face crop | Re-decoded per tier | Shared once, passed directly to emotion |
| Orchestrator pending flags | `_tier2_pending + _tier3_pending` | Single `_worker_pending` |
| Hardware support | Single implicit profile | Auto-detected: `edge_cpu` / `edge_gpu` / `workstation` / `cloud` |
| Gateway | No connection | Full ANP WebSocket node |
| Intelligence | None | CausalChain + AnomalyDetector + BehaviorModel |
| Context fields | 15 fields | 22 fields (+ depth, gaze, audio, action, causal, anomaly, behavior) |

### New Modules

| Module | Path | Status |
|--------|------|--------|
| ANP Client | `anp/client.py` | ✅ Complete |
| ANP Command Handler | `anp/command_handler.py` | ✅ Complete |
| Causal Chain | `intelligence/causal_chain.py` | ✅ Complete |
| Anomaly Detector | `intelligence/anomaly_detector.py` | ✅ Complete |
| Behavior Model | `intelligence/behavior_model.py` | ✅ Complete |
| Depth Estimator | `perception/depth/` | 🔧 Stub ready |
| Gaze Estimator | `perception/gaze/` | 🔧 Stub ready |
| Audio Channel | `perception/audio/` | 🔧 Stub ready |
| Action Recognizer | `perception/action/` | 🔧 Stub ready |

### New Config Sections

`hardware_profile`, `depth`, `gaze`, `audio`, `action_recognition`, `intelligence`, `gateway` — all optional, all default to `enabled: false`. Zero breaking changes on upgrade from v1.

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PERSEUS-Net v2.0 SYSTEM                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌────────────────────────────────────────────────────────────────────────┐
  │                        HARDWARE LAYER                                  │
  │   USB/CSI Camera · Microphone · Edge Compute · Output Peripherals      │
  └─────────────────────────────────┬──────────────────────────────────────┘
                                    │ Raw BGR @ 30 FPS + Audio stream
  ┌─────────────────────────────────▼──────────────────────────────────────┐
  │                      FRAME ACQUISITION                                 │
  │   CameraInputLayer (threaded, bounded queue)                           │
  │   FrameProcessor   (CLAHE, resize, quality gating)                     │
  │   AudioChannel     (Silero VAD → faster-whisper STT)  [optional]       │
  └─────────────────────────────────┬──────────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼──────────────────────────────────────┐
  │                  ADAPTIVE GATING ENGINE                                │
  │   Frame delta hash · Gesture state change · Face change · Audio inject │
  │   → Decides when to schedule UnifiedPerceptionWorker                   │
  └──────────────────┬──────────────────────────────────────────────────── ┘
                     │ trigger (adaptive — not fixed interval)
  ┌──────────────────▼──────────────────────────────────────────────────── ┐
  │             SYNC: Human Detection + Gesture (every frame)              │
  │   YOLOv8-nano Human Detector  |  MediaPipe Hands + Pose                │
  └──────────────────┬─────────────────────────────────────────────────────┘
                     │ if human_present → schedule worker
  ┌──────────────────▼─────────────────────────────────────────────────────┐
  │              UNIFIED PERCEPTION WORKER (async)                         │
  │                                                                        │
  │  Phase A [parallel futures]:                                           │
  │    ├── InsightFace ArcFace    — Face recognition + ByteTrack ID        │
  │    ├── YOLOv8-small           — Contextual object detection            │
  │    ├── L2CS-Net               — Gaze estimation (if enabled)           │
  │    └── MiDaS                  — Monocular depth (if enabled)           │
  │                                                                        │
  │  Phase B [sequential, uses Phase A face crop]:                         │
  │    ├── DeepFace EWA           — Emotion detection on shared crop       │
  │    └── MoViNet/SlowFast       — Temporal action recognition (if enabled│
  │                                                                        │
  │  Phase C [conditional — only if scene_interval elapsed]:               │
  │    └── Qwen-VL / LLaVA        — VLM scene analysis                     │
  └──────────────────┬─────────────────────────────────────────────────────┘
                     │ PerceptionState (all fields populated)
  ┌──────────────────▼─────────────────────────────────────────────────────┐
  │                      MEMORY SYSTEM                                     │
  │   TemporalMemoryBuffer  — rolling 30s snapshots, EWA trend analysis    │
  │   SessionMemory         — per-session activity/emotion/interaction log │
  │   ProfileStore          — persistent JSON user profiles                │
  └──────────────────┬─────────────────────────────────────────────────────┘
                     │ BufferSummary + SessionMemory
  ┌──────────────────▼─────────────────────────────────────────────────────┐
  │                    INTELLIGENCE LAYER                                  │
  │   CausalChain      — cause-effect pairs across interactions            │
  │   AnomalyDetector  — Isolation Forest + rule-based pattern detection   │
  │   BehaviorModel    — per-user pattern learning + predictive triggering │
  └──────────────────┬─────────────────────────────────────────────────────┘
                     │ causal_context + anomaly_notes + behavior_hints
  ┌──────────────────▼─────────────────────────────────────────────────────┐
  │                      CONTEXT ENGINE                                    │
  │   Synthesizes all signals → ContextObject (22 fields)                  │
  │   user_id · emotion · trend · valence · activity · gesture             │
  │   depth_cm · gaze_attention · audio_utterance · action_label           │
  │   scene · objects · event · causal_context · anomaly_notes             │
  └──────────────────┬─────────────────────────────────────────────────────┘
                     │ ContextObject
  ┌──────────────────▼─────────────────────────────────────────────────────┐
  │               PERSONALIZATION ENGINE                                   │
  │   UserProfile lookup → behavior instructions for LLM prompt            │
  │   EMA preference learning from interaction outcome feedback            │
  └──────────────────┬─────────────────────────────────────────────────────┘
                     │ ContextObject + BehaviorInstructions
          ┌──────────┴────────────┐
          │  gateway.enabled?     │
          ▼ YES                   ▼ NO (or offline)
  ┌───────────────────┐   ┌───────────────────────────────────────────────┐
  │   ANP CLIENT      │   │              AURA AGENT (local fallback)      │
  │   WebSocket node  │   │   LLM backend: Ollama | Anthropic | OpenAI    │
  │   → AURA Gateway  │   │   Structured JSON → AgentAction               │
  │   Gary + Skills   │   │   Cooldown enforcement per action type        │
  └────────┬──────────┘   └──────────────────┬────────────────────────────┘
           │ inbound commands                 │ AgentAction
           ▼                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                     BEHAVIOR EXECUTOR                                   │
  │   GatewayCommandHandler  → routes speak / led / config commands         │
  │   TTSEngine              → pyttsx3 / Coqui speech output                │
  │   LEDController          → NeoPixel ring patterns                       │
  │   MicrocontrollerBridge  → JSON serial to ESP32/Arduino                 │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Architecture

### Unified Perception Worker (v2.0)

v1 used three strictly separated tiers on fixed timers. v2 replaces this with a single `UnifiedPerceptionWorker` triggered adaptively:

```
Every frame (sync):
  detect_with_presence()  →  gesture.process()
  compute_frame_delta()   →  check_gating_conditions()
      │
      └─ triggers? → schedule UnifiedPerceptionWorker
                              │
               ┌──────────── Phase A (parallel) ─────────────┐
               │  face_recognizer.recognize()                │
               │  object_detector.detect()                   │
               │  gaze_estimator.estimate()      [if enabled │
               │  depth_estimator.estimate()     [if enabled]│
               └──────────── Phase B (sequential) ───────────┘
               │  emotion_detector.detect_emotion(face_crop) │
               │  action_recognizer.recognize(clip) [if enabl]
               └──────────── Phase C (conditional) ──────────┘
                  if (now - last_scene_ts) >= scene_interval:
                    vlm_analyzer.analyze_scene()
                    → fire on_perception_complete()
                    → build context → ANP or local agent
```

### Adaptive Gating

The worker is scheduled when **any** of these conditions is true:

| Trigger | Condition | Configurable |
|---------|-----------|-------------|
| Base interval | `elapsed >= perception_interval_seconds` | `pipeline.perception_interval_seconds` |
| Frame delta | Pixel change fraction > threshold | `pipeline.frame_delta_threshold` (default: 0.04) |
| Gesture change | Gesture label changed from last frame | `pipeline.gesture_change_trigger` |
| Face change | Different user_id recognized | `pipeline.face_change_trigger` |
| Audio injection | STT transcript received from AudioChannel | Automatic |

Minimum gap between triggers: 0.5s regardless of conditions.

---

## Module Reference

### Repository Structure

```
PERSEUS-Net/
│
├── main.py                              ← System entry point, full pipeline wiring
│
├── config/
│   ├── config.py                        ← Pydantic v2 typed settings + HardwareProfile auto-detect
│   └── settings.yaml                    ← Master configuration (all sections, all defaults)
│
├── perception/
│   ├── orchestrator.py                  ← Unified adaptive perception coordinator
│   ├── camera/
│   │   ├── capture.py                   ← Thread-safe camera acquisition (bounded queue)
│   │   └── processor.py                 ← CLAHE enhancement, resize, quality gating
│   ├── detection/
│   │   └── human_detector.py            ← YOLOv8 person detection + ByteTrack + spatial metadata
│   ├── recognition/
│   │   └── face_recognizer.py           ← InsightFace ArcFace + cosine similarity + enrollment DB
│   ├── emotion/
│   │   └── emotion_detector.py          ← DeepFace + EWA smoothing + body orientation gating
│   ├── gesture/
│   │   └── gesture_recognizer.py        ← MediaPipe Hands + Pose + waving detection
│   ├── objects/
│   │   └── object_detector.py           ← COCO-class detection + activity inference rules
│   ├── scene/
│   │   └── vlm_analyzer.py              ← VLM scene understanding (Qwen-VL/LLaVA/MiniCPM-V)
│   ├── depth/                           ← MiDaS monocular depth [stub — install to activate]
│   ├── gaze/                            ← L2CS-Net gaze estimation [stub — install to activate]
│   ├── audio/                           ← Silero VAD + faster-whisper STT [stub — install to activate]
│   └── action/                          ← MoViNet temporal action recognition [stub — install to activate]
│
├── intelligence/
│   ├── causal_chain.py                  ← Rolling cause-effect interaction history
│   ├── anomaly_detector.py              ← Isolation Forest + rule-based behavioral anomaly detection
│   └── behavior_model.py               ← Per-user pattern learning + predictive trigger scoring
│
├── anp/
│   ├── client.py                        ← ANP WebSocket node (asyncio, auto-reconnect, heartbeat)
│   └── command_handler.py               ← Inbound AURA gateway command → BehaviorExecutor routing
│
├── memory/
│   ├── temporal_buffer.py               ← Rolling 30s PerceptionSnapshot buffer + trend analysis
│   └── session_memory.py                ← Per-session activity/emotion/interaction log
│
├── context/
│   └── context_engine.py                ← All signals → ContextObject (22 fields)
│
├── personalization/
│   ├── user_profile.py                  ← UserProfile dataclass + JSON persistence
│   └── personalization_engine.py        ← EMA preference learning + prompt generation
│
├── agent/
│   └── aura_agent.py                    ← Local LLM agent fallback (Ollama/Anthropic/OpenAI)
│
├── behavior/
│   ├── executor.py                      ← Action execution coordinator
│   ├── tts/
│   │   └── tts_engine.py                ← pyttsx3 + Coqui TTS with tone-adaptive rate
│   └── leds/
│       └── led_controller.py            ← NeoPixel LED ring control + patterns
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

## Intelligence Layer

### CausalChain (`intelligence/causal_chain.py`)

Tracks cause-effect pairs across every interaction. Each entry records the full state at trigger time (emotion, gesture, activity, event) and the action taken. After the next perception cycle, the emotional response is recorded as outcome.

```
Cause:  emotion=frustrated · activity=working_at_computer · gesture=fist
Action: greet → "Hey, need a break?"
Effect: emotion=neutral · outcome=positive
```

Over multiple sessions, `get_pattern_hints()` surfaces behavioral rules like:
> *"Avoid 'greet' when user is 'working_at_computer' — consistently negative"*

These hints are injected directly into the LLM system prompt.

### AnomalyDetector (`intelligence/anomaly_detector.py`)

Dual approach: rule-based checks run every snapshot, IsolationForest runs when 50+ samples are available.

| Anomaly | Detection Method | Severity |
|---------|-----------------|----------|
| User motionless > 20 min | Rule: last gesture change timestamp | Medium |
| Sustained negative emotion (75%+ of recent) | Rule: emotion history ratio | High |
| Rapid gesture oscillation (≥8 changes in 10 frames) | Rule: change counter | Low |
| Repeated unknown face (≥3 appearances) | Rule: unrecognized count | Medium |
| Statistical outlier behavior | IsolationForest (sklearn optional) | Low |

Anomaly descriptions are injected as `anomaly_notes` into every `ContextObject`.

### BehaviorModel (`intelligence/behavior_model.py`)

Pure statistical tracking — no ML libraries required. Learns per-user:

- **Arrival hour distribution** — which hours the user typically appears
- **Emotion timeline** — what emotion is typical at 15-min session intervals (e.g., mood drops after 60 min)
- **Gesture precursors** — which gestures precede interactions most often
- **Interaction likelihood score** — 0.0–1.0 prediction of whether an interaction is imminent

The likelihood score enables **predictive pre-warming**: models are loaded before the user actually triggers, reducing response latency from 2–3s toward near-instant.

---

## ANP Gateway Integration

PERSEUS-Net v2.0 can register as an authenticated node to the AURA gateway, giving it access to the full skill ecosystem (Home Assistant, Spotify, Google Calendar, trading bots, Notion, etc.) without any local skill implementation.

### How It Works

```
PERSEUS-Net                          AURA Gateway
    │                                     │
    │── HELLO {node_id, token, caps} ────▶│
    │◀─ WELCOME {session_id} ─────────────│
    │                                     │
    │── ANPEvent (perception payload) ───▶│
    │   {content, meta: {emotion,         │
    │    gesture, activity, depth_cm,     │  Gary (orchestrator)
    │    gaze_at_device, utterance...}}   │  → matches workflow / intent
    │                                     │  → spawns skill sub-agents
    │                                     │  → executes Home Assistant / Spotify / etc.
    │◀─ ANPEvent {action_type, message} ──│
    │   {speak: "Playing your focus       │
    │    playlist", led_pattern: "pulse"} │
    │                                     │
    │── PING ────────────────────────────▶│
    │◀─ PONG ─────────────────────────────│
```

### Structured Perception Event

Every `on_perception_complete` sends a typed event to the gateway:

```json
{
  "type": "utterance",
  "node_id": "perseus_node_001",
  "source": "perception",
  "content": "User: Alex (98%)\nEmotion: happy (positive, trend: improving)\nActivity: working_at_computer\nGaze: at device\nDistance: 65cm",
  "meta": {
    "user_id": "user_001",
    "emotion": "happy",
    "valence": "positive",
    "gesture": "thumbs_up",
    "activity": "working_at_computer",
    "gaze_at_device": true,
    "depth_cm": 65.0,
    "event": "user_approval_gesture",
    "notable_events": [],
    "audio_utterance": null
  }
}
```

### Inbound Commands

The gateway sends back typed commands that `GatewayCommandHandler` routes to local executors:

| Command | Handler | Example |
|---------|---------|---------|
| `speak` | TTSEngine | `{"action_type": "speak", "message": "Your standup starts in 5 min", "tone": "warm"}` |
| `led_pattern` | LEDController | `{"action_type": "led_pattern", "pattern": "pulse_blue"}` |
| `silence` | (no-op) | Gateway chose not to respond |
| `config_update` | Config hook | `{"key": "perception_interval", "value": 2.0}` |

### Enabling the Gateway

```yaml
# config/settings.yaml
gateway:
  enabled: true
  host: "your-aura-host"
  port: 8765
  node_id: "perseus_node_001"
  token: ""                    # copy from ~/.aura/nodes.yaml on the gateway
  caps: ["vision", "gesture", "emotion", "audio", "depth"]
  reconnect_interval_seconds: 5.0
  heartbeat_interval_seconds: 30.0
  local_agent_fallback: true   # use local agent if gateway is unreachable
```

Or via environment variables:
```bash
GATEWAY_ENABLED=true
GATEWAY_HOST=192.168.1.100
GATEWAY_TOKEN=your-token-here
GATEWAY_NODE_ID=perseus_node_001
```

---

## Hardware Profiles

PERSEUS-Net v2.0 auto-detects hardware at startup and applies the appropriate profile. Override with `--profile` flag or `hardware_profile.mode` in config.

| Profile | Hardware | Auto-detect Condition |
|---------|----------|----------------------|
| `edge_cpu` | Raspberry Pi, ARM SBCs | ARM architecture or < 16GB RAM, no CUDA |
| `edge_gpu` | Jetson Nano/Orin/Xavier | CUDA available + Jetson GPU name detected |
| `workstation` | PC/Mac with GPU | CUDA available + ≥ 16GB RAM |
| `cloud` | Remote inference | Set manually |

### Profile Behavior

| Setting | edge_cpu | edge_gpu | workstation |
|---------|----------|----------|-------------|
| `perception_interval` | ≥ 5.0s | 3.0s | 3.0s |
| `scene_interval` | ≥ 20.0s | 10.0s | 10.0s |
| `max_worker_threads` | 2 | 4 | 4+ |
| Detection model | yolov8n.pt | yolov8n.pt (cuda) | yolov8s.pt (cuda) |
| Depth enabled | ❌ | ✅ MiDaS_small | ✅ DPT_Hybrid |
| Gaze enabled | ❌ | ❌ | ✅ |
| Action recognition | ❌ | ❌ | configurable |

---

## AI Models

### Vision Models (Core)

| Model | Variant | Purpose | Tier |
|-------|---------|---------|------|
| **YOLOv8** | nano (detect) | Human presence + spatial zones | Sync/frame |
| **YOLOv8** | small (objects) | 24-class context object detection | Worker Phase A |
| **InsightFace** | buffalo_sc / buffalo_l | ArcFace embedding + face detection | Worker Phase A |
| **DeepFace** | FER+ / AffectNet | 7-class emotion + EWA smoothing | Worker Phase B |
| **MediaPipe** | Hands + Pose | Gesture classification + body orientation | Sync/frame |

### Vision Models (Optional)

| Model | Purpose | Profile Requirement |
|-------|---------|---------------------|
| **MiDaS** | Monocular depth → cm estimate | edge_gpu+ |
| **L2CS-Net** | Gaze yaw/pitch → at-device detection | workstation |
| **MoViNet A0** / SlowFast | Temporal action recognition | workstation |

### Language Models

| Backend | Model | Privacy | Notes |
|---------|-------|---------|-------|
| **Ollama** | llama3.1:8b | 100% local | Default, no data leaves device |
| **Ollama** | llama3.2:3b | 100% local | Low-RAM option |
| **Anthropic** | claude-haiku-4-5-20251001 | Text only to cloud | Fast, cheap |
| **OpenAI** | gpt-4o-mini | Text only to cloud | |
| **AURA Gateway** | Gary → any configured LLM | Via gateway | Full skill access |

### Vision-Language Models (Phase C, Optional)

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
- Ollama for local LLM — [ollama.com](https://ollama.com)
- CUDA 11.8+ optional for GPU acceleration

### Step 1 — Clone

```bash
git clone https://github.com/Sudharsanselvaraj/PERSEUS-Net-Perception-Environment-Reasoning-System.git
cd PERSEUS-Net-Perception-Environment-Reasoning-System
```

### Step 2 — Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install Dependencies (Tiered)

```bash
# Core vision — fast (~2 min)
pip install opencv-python numpy Pillow ultralytics mediapipe
pip install pydantic pydantic-settings python-dotenv PyYAML
pip install loguru rich click pyserial httpx

# Face + emotion (~5 min)
pip install insightface onnxruntime scikit-learn
pip install deepface tf-keras

# ANP gateway integration
pip install websockets

# LLM agent — choose one or more
pip install httpx               # Ollama (no extra install needed)
pip install anthropic           # Claude API
pip install openai              # OpenAI API

# VLM scene analysis — optional, heavy (~10 min)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate sentencepiece

# Optional advanced modules
pip install timm                # MiDaS depth
pip install faster-whisper      # STT audio channel
# L2CS-Net gaze: install from https://github.com/Ahmednull/L2CS-Net
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
# Agent backend
AURA_AGENT_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Camera
CAMERA_DEVICE_ID=0

# Debug
AURA_DEBUG=false
AURA_LOG_LEVEL=INFO

# Hardware
SERIAL_ENABLED=false

# Gateway (optional — leave blank to use local agent)
GATEWAY_ENABLED=false
GATEWAY_HOST=localhost
GATEWAY_PORT=8765
GATEWAY_TOKEN=
GATEWAY_NODE_ID=perseus_node_001

# Cloud LLM keys (only if using cloud backends)
ANTHROPIC_API_KEY=
```

### Step 5 — Pull Ollama Model

```bash
ollama pull llama3.1:8b        # Standard (~4.7 GB)
# or
ollama pull llama3.2:3b        # Low-RAM (~2 GB)
```

### Step 6 — Enroll First User

```bash
python scripts/enroll.py
```

---

## Configuration

All configuration is in `config/settings.yaml`. Environment variables in `.env` override YAML values at runtime.

### Hardware Profile

```yaml
hardware_profile:
  mode: "auto"          # auto | edge_cpu | edge_gpu | workstation | cloud
  force_cpu_modules: [] # e.g. ["vlm", "depth"] to force specific modules to CPU
```

### Camera

```yaml
camera:
  device_id: 0
  width: 1280
  height: 720
  fps: 30
  backend: "dshow"      # dshow (Windows) | v4l2 (Linux) | avfoundation (macOS)
```

### Pipeline (Adaptive Gating)

```yaml
pipeline:
  perception_interval_seconds: 3.0    # base interval between worker runs
  scene_interval_seconds: 10.0        # min seconds between VLM calls
  max_worker_threads: 4
  frame_delta_threshold: 0.04         # pixel change fraction to trigger early
  gesture_change_trigger: true        # trigger on gesture state change
  face_change_trigger: true           # trigger on new face detected
```

### Intelligence Layer

```yaml
intelligence:
  causal_chain_depth: 10              # how many cause-effect pairs to keep
  anomaly_window_minutes: 15          # lookback window for anomaly detection
  anomaly_contamination: 0.05         # isolation forest expected anomaly rate
  behavioral_pattern_min_sessions: 3  # sessions before predictions activate
  predictive_trigger_enabled: true    # pre-warm on high interaction likelihood
```

### Optional Modules

```yaml
depth:
  enabled: false
  model_type: "MiDaS_small"           # MiDaS_small | DPT_Hybrid | DPT_Large
  device: "cpu"

gaze:
  enabled: false
  model_path: "models/L2CSNet_gaze360.pkl"
  attention_threshold_deg: 20.0       # angle within which user is "looking at device"

audio:
  enabled: false
  stt_model: "base"                   # tiny | base | small | medium
  wake_words: ["hey aura", "aura"]    # empty = always listen
  vad_threshold: 0.5
  silence_duration_ms: 700

action_recognition:
  enabled: false
  model: "movinet_a0"                 # movinet_a0 | slowfast_r50
  clip_length_frames: 16
  inference_interval_frames: 32
```

### Agent (Local Fallback)

```yaml
agent:
  backend: "ollama"                   # ollama | anthropic | openai
  ollama_model: "llama3.1:8b"
  anthropic_model: "claude-haiku-4-5-20251001"
  openai_model: "gpt-4o-mini"
  max_tokens: 512
  temperature: 0.7
  timeout_seconds: 10.0
```

### Cooldowns (seconds)

```yaml
cooldowns:
  greet: 300
  reminder: 600
  comment: 180
  question: 300
  reaction: 30
  silence: 0
```

---

## Usage

### Running PERSEUS-Net

```bash
# Standard run (uses auto hardware profile, local agent)
python main.py

# With live annotated debug window
python main.py --show-video

# Override hardware profile
python main.py --profile workstation
python main.py --profile edge_cpu

# Disable VLM (Phase C) for faster startup
python main.py --no-vlm

# Perception only — no TTS or LEDs
python main.py --dry-run

# Custom config
python main.py --config my_config.yaml

# Face enrollment wizard
python main.py --enroll
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config (default: config/settings.yaml) |
| `--enroll` | Run interactive face enrollment wizard |
| `--no-vlm` | Disable VLM scene analysis (Phase C) |
| `--show-video` | Show live debug video with all perception overlays |
| `--dry-run` | Skip TTS and LED output |
| `--profile MODE` | Override hardware profile: edge_cpu / edge_gpu / workstation / cloud |

### Debug Video Overlays (v2.0)

When running with `--show-video`, the debug window shows:

- Green bounding boxes for all detected persons
- Distance label in cm (if depth enabled)
- Emotion + valence
- Gesture + body orientation
- Gaze indicator (AT DEVICE / elsewhere)
- Temporal action label (if action recognition enabled)
- ANP gateway connection status (CONNECTED / DISCONNECTED)

### User Management

```bash
python scripts/enroll.py                             # Interactive enrollment
python scripts/enroll.py --user-id u1 --name "Alex" --frames 12
python scripts/list_users.py                         # Show all enrolled profiles
python scripts/test_camera.py                        # Camera diagnostic
```

---

## Hardware Integration

### Microcontroller Serial Protocol

PERSEUS-Net communicates with ESP32 or Arduino over USB serial using newline-delimited JSON:

```json
{"cmd": "led_color", "r": 255, "g": 100, "b": 50}
{"cmd": "led_pattern", "pattern": "pulse_blue"}
{"cmd": "servo_move", "yaw": 90, "pitch": 75}
{"cmd": "display_text", "text": "Hello!", "color": "white"}
{"cmd": "play_sound", "file": "chime.wav"}
```

Gateway-originated LED commands are automatically forwarded to the serial bridge via `GatewayCommandHandler`.

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
  serial_port: "COM3"              # Windows
  # serial_port: "/dev/ttyUSB0"   # Linux
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
| User profiles | `data/profiles/*.json` | Preferences, interaction stats, behavioral patterns |
| Session logs | `data/sessions/sessions.db` | Activity, emotion, interaction history |
| System logs | `data/logs/` | Timestamped pipeline logs |

### Privacy Architecture

- All camera and audio processing is **fully on-device** — no video or audio is transmitted
- When using Ollama backend, **no data ever leaves the machine**
- When using Anthropic/OpenAI, only the **text context string** is sent — never images, never audio, never biometric vectors
- When using AURA gateway, the **text perception event** is sent over your local network — the gateway never receives raw frames
- Face embeddings are 512-dimensional mathematical vectors — raw face images are never stored

### API Key Security

```bash
echo ".env" >> .gitignore
git rm --cached .env 2>/dev/null
```

---

## Troubleshooting

### Camera Not Opening

```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    print(f'Device {i}:', cap.isOpened())
    cap.release()
"
```

Set correct `device_id` and `backend: "dshow"` (Windows) in settings.yaml.

### ANP Gateway Not Connecting

```bash
# Verify gateway is running
curl http://your-gateway-host:3001/health

# Check token matches nodes.yaml on gateway
# Check node_id is registered

# Test WebSocket manually
python -c "
import asyncio, websockets, json
async def test():
    async with websockets.connect('ws://localhost:8765/anp') as ws:
        await ws.send(json.dumps({'type':'hello','node_id':'test','token':'','caps':[]}))
        print(await ws.recv())
asyncio.run(test())
"
```

### MediaPipe Protobuf Conflict

```bash
pip uninstall mediapipe protobuf -y
pip install protobuf==4.25.3 mediapipe==0.10.14
```

### Ollama Returns 500 Error

```bash
ollama list                    # verify model is downloaded
ollama serve                   # restart service
ollama pull llama3.2:3b        # switch to smaller model if low RAM
```

### Optional Module Not Loading

All optional modules (depth, gaze, audio, action) fail gracefully — the system continues without them. Check logs for `WARNING: [module] module unavailable: [reason]`. Install the required package and re-run.

### Slow Performance

| Symptom | Fix |
|---------|-----|
| Low FPS | Set `scene.enabled: false`, use `yolov8n.pt` |
| High RAM | Use `llama3.2:3b`, set `use_quantization: true` |
| Slow emotion | Set `emotion.backend: "opencv"` |
| Agent timeout | Increase `agent.timeout_seconds: 30.0` |
| Too frequent | Increase `pipeline.perception_interval_seconds` |
| ANP lag | Check `gateway.heartbeat_interval_seconds` |

### Enable Full Debug Logging

```bash
# Windows PowerShell
$env:AURA_DEBUG="true"
$env:AURA_LOG_LEVEL="DEBUG"
python main.py --show-video --profile workstation
```

---

## Development

### Running Tests

```bash
pytest                               # all tests
pytest tests/unit/                   # unit only
pytest -k "test_agent"               # specific module
pytest --cov=. --cov-report=html     # HTML coverage report
```

### Adding a New Perception Module

1. Create `perception/<module>/<module>.py`
2. Implement `initialize()` and a `process(frame)` method returning a typed dataclass
3. Add config model to `config/config.py` and section to `config/settings.yaml`
4. Load conditionally in `main.py → _load_optional_modules()`
5. Pass to `PerceptionOrchestrator.__init__()` as optional parameter
6. Call in `_run_unified_worker()` Phase A or B with exception isolation
7. Add field to `PerceptionState` and `ContextObject`
8. Include in `context/context_engine.py → build_context()`

### Adding a New Intelligence Module

1. Create `intelligence/<module>.py`
2. Instantiate in `main.py → build_components()`
3. Pass into `_on_perception_complete()` → `context_engine.build_context()`
4. Add config section to `intelligence:` in `settings.yaml`

### Implementing a Stub Module (depth / gaze / audio / action)

Each stub directory has an `__init__.py`. Create the implementation file:

```
perception/depth/depth_estimator.py    → class DepthEstimator: initialize(), estimate_primary_distance(frame) → float
perception/gaze/gaze_estimator.py      → class GazeEstimator:  initialize(), estimate(frame) → GazeResult(at_device: bool)
perception/audio/audio_channel.py      → class AudioChannel:   start(callback), stop()
perception/action/action_recognizer.py → class ActionRecognizer: initialize(), recognize(clip) → ActionResult(label: str)
```

### Switching LLM Backend at Runtime

```bash
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
- [MiDaS](https://github.com/isl-org/MiDaS) — Monocular depth estimation
- [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) — Gaze estimation
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — STT transcription
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice activity detection
- [Ollama](https://ollama.com) — Local LLM inference runtime
- [Anthropic Claude](https://anthropic.com) — Cloud LLM backend
- [AURA Gateway](.) — Multi-agent skill execution platform

---

*PERSEUS-Net v2.0 — Perception · Environment · Reasoning · Understanding · System*
