# AI Companion Device — Complete Technical Architecture

> **Document Type:** Technical Architecture Specification  
> **Version:** 1.0  
> **Audience:** Software Engineers, ML Engineers, Embedded Systems Engineers  
> **Status:** Draft for Implementation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Vision Perception Pipeline](#3-vision-perception-pipeline)
4. [Context and Memory System](#4-context-and-memory-system)
5. [Context Engine](#5-context-engine)
6. [Personalization Engine](#6-personalization-engine)
7. [AI Agent Layer](#7-ai-agent-layer)
8. [Assistant Behavior Layer](#8-assistant-behavior-layer)
9. [System Performance Strategy](#9-system-performance-strategy)
10. [Hardware Architecture](#10-hardware-architecture)
11. [Data Flow Diagram](#11-data-flow-diagram)
12. [Example Context Outputs](#12-example-context-outputs)
13. [Future Improvements](#13-future-improvements)

---

## 1. Project Overview

### 1.1 Purpose

The AI Companion Device is a stationary or semi-mobile smart assistant that observes its physical environment through a camera, understands the human user's presence, activities, and emotional state, and responds with intelligent, context-aware behavior. Unlike traditional voice assistants that are purely reactive, this device is continuously aware — it perceives without being asked, builds a model of the user's world, and proactively engages in ways that feel natural, empathetic, and useful.

The device is not a full robot with locomotion. It is a personalized AI agent embedded in a physical form factor — a companion that can see, understand, remember, and respond. Think of it as an always-on AI entity that shares your space and knows your habits, mood, and needs over time.

### 1.2 Core Goals

| Goal | Description |
|------|-------------|
| **Ambient Awareness** | Continuously observe the environment without requiring the user to initiate interaction |
| **Human-Centric Perception** | Accurately identify who the user is, how they feel, and what they are doing |
| **Contextual Intelligence** | Combine visual, temporal, and historical signals to understand the full situation |
| **Proactive Engagement** | Initiate interactions, reminders, and responses based on context — not just commands |
| **Personalization** | Adapt behavior to each user's preferences, routines, and interaction patterns |
| **Real-Time Performance** | Maintain smooth, low-latency operation on edge hardware without cloud dependency |
| **Privacy-First Design** | Process all sensitive data on-device; minimize or eliminate cloud data transmission |

### 1.3 Key Use Cases

- **Home companion**: Greeting family members, reminding them of tasks, noticing when someone looks tired or stressed
- **Productivity assistant**: Detecting focused work sessions vs. distraction, offering time reminders, recognizing meeting context
- **Elderly care aid**: Monitoring activity patterns, detecting falls or unusual behavior, providing companionship
- **Personal wellness tracker**: Reading emotional signals over time, encouraging breaks, celebrating positive milestones
- **Smart home hub integration**: Using scene understanding to trigger smart home actions based on observed context

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture

The system is organized into six major layers that transform raw camera input into intelligent assistant behavior:

```
┌──────────────────────────────────────────────────────────────────┐
│                        PHYSICAL LAYER                            │
│        Camera Sensor → Edge Compute Device → Output Hardware     │
└──────────────────────────┬───────────────────────────────────────┘
                           │ Raw Video Frames
┌──────────────────────────▼───────────────────────────────────────┐
│                   VISION PERCEPTION PIPELINE                     │
│   Frame Pre-processing → Detection → Recognition → Understanding │
└──────────────────────────┬───────────────────────────────────────┘
                           │ Structured Perception Outputs
┌──────────────────────────▼───────────────────────────────────────┐
│                  CONTEXT AND MEMORY SYSTEM                       │
│      Temporal Buffers → Context Aggregation → Memory Store       │
└──────────────────────────┬───────────────────────────────────────┘
                           │ Unified Context Object
┌──────────────────────────▼───────────────────────────────────────┐
│                      CONTEXT ENGINE                              │
│        Event Detection → Context Structuring → State Updates     │
└──────────────────────────┬───────────────────────────────────────┘
                           │ Context + Personalization Signals
┌──────────────────────────▼───────────────────────────────────────┐
│                    PERSONALIZATION ENGINE                        │
│        User Profile → Preference Learning → Behavior Adaptation  │
└──────────────────────────┬───────────────────────────────────────┘
                           │ Enriched Context + Behavioral Instructions
┌──────────────────────────▼───────────────────────────────────────┐
│                      AI AGENT LAYER (AURA)                       │
│      LLM-based Decision Making → Action Selection → Response Gen │
└──────────────────────────┬───────────────────────────────────────┘
                           │ Action Commands
┌──────────────────────────▼───────────────────────────────────────┐
│                  ASSISTANT BEHAVIOR LAYER                        │
│        TTS → Display → Animation → Notifications → Actuators     │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Summary

1. The **camera** captures continuous video at a configured frame rate (e.g., 15–30 FPS).
2. The **Vision Perception Pipeline** processes each frame (or a subset) through a sequence of ML models to extract: who is present, what they are doing, how they feel, what gestures they are making, and what objects are visible.
3. Perception outputs are accumulated in the **Context and Memory System**, which maintains rolling temporal buffers and aggregates observations into a coherent picture of the user's state.
4. The **Context Engine** synthesizes buffer data into a structured context object — a machine-readable description of the current situation.
5. The **Personalization Engine** enriches the context with user-specific knowledge: known preferences, behavioral patterns, and interaction history.
6. The **AI Agent (Aura)** receives the enriched context and decides what action to take — or whether to take no action at all.
7. The **Assistant Behavior Layer** executes the chosen action through output modalities: voice, display, light, sound, or other actuators.

### 2.3 Processing Modes

The system operates in three concurrent processing modes to balance latency and cost:

| Mode | Trigger | Frequency | Models Involved |
|------|---------|-----------|-----------------|
| **Realtime** | Every frame | 15–30 FPS | YOLOv8-nano, MediaPipe, Face detection |
| **Periodic** | Every N seconds | 2–10s | Emotion model, Face recognition, Scene VLM |
| **Event-driven** | On state change | On demand | Full context build, Agent LLM call |

---

## 3. Vision Perception Pipeline

The Vision Perception Pipeline is the sensory system of the companion device. It transforms raw pixel data into structured, semantic information about the user and their environment.

### 3.1 Camera Input Layer

**Purpose:** Acquire raw video frames and prepare them for processing.

**Implementation Details:**

The camera input layer handles device initialization, frame capture, and hardware-level configuration. The system uses OpenCV's `VideoCapture` interface as the primary acquisition method, with support for USB cameras, CSI cameras (on Jetson), and IP cameras.

```python
import cv2
import threading
from queue import Queue
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: int = cv2.CAP_V4L2          # Use CAP_DSHOW on Windows
    buffer_size: int = 1                  # Minimize internal buffer lag

class CameraInputLayer:
    """
    Thread-safe camera capture layer with hardware-level configuration.
    Uses a dedicated capture thread to prevent frame drops in downstream processing.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue = Queue(maxsize=2)   # Bounded queue prevents memory accumulation
        self._running = False
        self._capture_thread = None

    def initialize(self) -> bool:
        self.cap = cv2.VideoCapture(self.config.device_id, self.config.backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        return self.cap.isOpened()

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                # Drop oldest frame if queue is full (always serve freshest frame)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Exception:
                        pass
                self.frame_queue.put(frame)

    def start(self):
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=0.1)
        except Exception:
            return None

    def stop(self):
        self._running = False
        if self.cap:
            self.cap.release()
```

**Camera Hardware Recommendations:**

| Use Case | Camera Model | Resolution | Interface |
|----------|-------------|------------|-----------|
| Desktop companion | Logitech BRIO 4K | 4K / 1080p | USB 3.0 |
| Jetson embedded | IMX219 / IMX477 CSI | 1080p | CSI-2 |
| Wide-angle room monitoring | 180° fisheye USB cam | 1080p | USB 2.0 |
| Outdoor / strong lighting | Global shutter industrial | 720p+ | USB 3.0 |

**Key Design Decisions:**
- Use `buffer_size=1` to always process the most recent frame, not buffered backlog
- Run capture in a separate thread to decouple acquisition from processing
- Use `MJPEG` compression at the hardware level to reduce USB bandwidth

---

### 3.2 Frame Processing (OpenCV)

**Purpose:** Pre-process raw frames to normalize input for downstream ML models, reduce computational load, and extract metadata.

**Processing Steps:**

```python
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class FrameMetadata:
    timestamp: float
    frame_id: int
    original_shape: tuple
    processed_shape: tuple
    brightness: float
    blur_score: float
    is_usable: bool

class FrameProcessor:
    """
    Applies image pre-processing to raw camera frames.
    Normalizes size, color space, exposure, and filters unusable frames.
    """

    TARGET_WIDTH = 640
    TARGET_HEIGHT = 480
    BLUR_THRESHOLD = 80.0       # Laplacian variance threshold
    MIN_BRIGHTNESS = 20         # Minimum mean pixel value (0-255)
    MAX_BRIGHTNESS = 240

    def process(self, frame: np.ndarray, frame_id: int, timestamp: float):
        original_shape = frame.shape

        # Step 1: Resize to standard processing resolution
        resized = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT),
                             interpolation=cv2.INTER_LINEAR)

        # Step 2: Convert to RGB (OpenCV default is BGR)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Improves detection in low/uneven lighting
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Step 4: Compute quality metrics
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))

        # Step 5: Determine if frame is usable
        is_usable = (
            blur_score >= self.BLUR_THRESHOLD and
            self.MIN_BRIGHTNESS <= brightness <= self.MAX_BRIGHTNESS
        )

        metadata = FrameMetadata(
            timestamp=timestamp,
            frame_id=frame_id,
            original_shape=original_shape,
            processed_shape=resized.shape,
            brightness=brightness,
            blur_score=blur_score,
            is_usable=is_usable
        )

        return enhanced_bgr, rgb_frame, metadata
```

**Why CLAHE?** Adaptive histogram equalization locally enhances contrast across different regions of the image, making it effective for mixed lighting conditions (e.g., a user lit from one side by a window). This directly improves detection accuracy for all downstream models.

**Frame Quality Gating:** Frames that are too blurry (motion blur, camera shake) or too dark/bright are flagged as unusable and skipped by the perception pipeline. This prevents low-quality inputs from corrupting downstream model outputs and the memory system.

---

### 3.3 Human Detection (YOLOv8)

**Purpose:** Detect the presence and location of human figures in each frame, providing bounding boxes and confidence scores.

**Model:** YOLOv8-nano or YOLOv8-small (Ultralytics)

**Why YOLOv8?**
- YOLOv8 provides an excellent accuracy/speed tradeoff for edge deployment
- The `nano` variant runs at 60+ FPS on modern CPUs and 200+ FPS on Jetson
- Built-in support for ONNX, TensorRT, and CoreML export
- Superior small-object detection compared to earlier YOLO versions
- Can be filtered to detect only the `person` class (COCO class 0), reducing compute

```python
from ultralytics import YOLO
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class HumanDetection:
    bbox: tuple           # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    center: tuple         # (cx, cy)
    area: int             # Bounding box area in pixels
    relative_position: str  # "left", "center", "right", "close", "far"

class HumanDetector:
    """
    YOLOv8-based human presence detector.
    Filters detections to persons only and computes spatial metadata.
    """

    PERSON_CLASS_ID = 0
    CONFIDENCE_THRESHOLD = 0.45
    CLOSE_AREA_THRESHOLD = 100_000  # Pixel area threshold for "close" proximity

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        # Export to ONNX for faster inference if not already done
        # self.model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)

    def detect(self, frame: np.ndarray) -> List[HumanDetection]:
        results = self.model(frame, classes=[self.PERSON_CLASS_ID],
                             conf=self.CONFIDENCE_THRESHOLD, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                frame_width = frame.shape[1]

                # Compute relative spatial position
                if cx < frame_width * 0.33:
                    h_pos = "left"
                elif cx > frame_width * 0.66:
                    h_pos = "right"
                else:
                    h_pos = "center"

                proximity = "close" if area > self.CLOSE_AREA_THRESHOLD else "far"
                relative_position = f"{proximity}_{h_pos}"

                detections.append(HumanDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    center=(cx, cy),
                    area=area,
                    relative_position=relative_position
                ))

        return detections
```

**Detection Optimization Notes:**
- Use `half=True` (FP16) inference on CUDA GPUs for 2× speedup
- Export to TensorRT on Jetson for maximum throughput
- Filter to `classes=[0]` (person only) to skip irrelevant detections
- For multi-person households, track individuals across frames using ByteTrack (built into Ultralytics)

---

### 3.4 Face Recognition (InsightFace / ArcFace)

**Purpose:** Identify which registered user is present by matching detected faces against a stored face database.

**Model:** InsightFace with ArcFace backbone (buffalo_l or buffalo_sc for lightweight deployment)

**Why ArcFace / InsightFace?**
- ArcFace achieves state-of-the-art verification accuracy on LFW and other benchmarks
- InsightFace provides a complete pipeline: detection, alignment, embedding extraction
- Face embeddings (512-dim vectors) enable fast cosine similarity matching
- Robust to lighting variation, partial occlusion, and moderate pose changes
- buffalo_sc (small/compact) runs efficiently on CPU for edge devices

```python
import insightface
import numpy as np
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class FaceRecognitionResult:
    user_id: Optional[str]          # None if unrecognized
    display_name: Optional[str]
    confidence: float               # Cosine similarity score (0–1)
    face_bbox: tuple
    face_embedding: np.ndarray
    is_new_face: bool               # True if never seen before

class FaceRecognitionSystem:
    """
    InsightFace-based face recognition system with enrolled user database.
    Uses ArcFace embeddings and cosine similarity for identity verification.
    """

    RECOGNITION_THRESHOLD = 0.45   # Cosine similarity cutoff for positive match
    DATABASE_PATH = "data/face_db.pkl"

    def __init__(self):
        # Initialize InsightFace model (buffalo_sc for edge, buffalo_l for accuracy)
        self.app = insightface.app.FaceAnalysis(name='buffalo_sc',
                                                 allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(320, 320))   # ctx_id=-1 for CPU
        self.face_database: Dict[str, Dict] = self._load_database()

    def _load_database(self) -> Dict:
        if Path(self.DATABASE_PATH).exists():
            with open(self.DATABASE_PATH, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_database(self):
        with open(self.DATABASE_PATH, 'wb') as f:
            pickle.dump(self.face_database, f)

    def enroll_user(self, user_id: str, display_name: str,
                    enrollment_frames: List[np.ndarray]) -> bool:
        """
        Register a new user by computing average embedding from multiple frames.
        More enrollment frames = more robust identity representation.
        """
        embeddings = []
        for frame in enrollment_frames:
            faces = self.app.get(frame)
            if faces:
                embeddings.append(faces[0].embedding)

        if not embeddings:
            return False

        # Store average embedding for robustness
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding)   # L2 normalize

        self.face_database[user_id] = {
            "embedding": avg_embedding,
            "display_name": display_name,
            "enrollment_count": len(embeddings)
        }
        self._save_database()
        return True

    def recognize(self, frame: np.ndarray) -> List[FaceRecognitionResult]:
        faces = self.app.get(frame)
        results = []

        for face in faces:
            query_embedding = face.embedding / np.linalg.norm(face.embedding)
            best_match_id = None
            best_match_name = None
            best_score = 0.0

            for user_id, data in self.face_database.items():
                score = float(cosine_similarity(
                    query_embedding.reshape(1, -1),
                    data["embedding"].reshape(1, -1)
                )[0][0])

                if score > best_score:
                    best_score = score
                    best_match_id = user_id
                    best_match_name = data["display_name"]

            is_recognized = best_score >= self.RECOGNITION_THRESHOLD
            bbox = tuple(face.bbox.astype(int))

            results.append(FaceRecognitionResult(
                user_id=best_match_id if is_recognized else None,
                display_name=best_match_name if is_recognized else None,
                confidence=best_score,
                face_bbox=bbox,
                face_embedding=query_embedding,
                is_new_face=(not is_recognized and best_score < 0.2)
            ))

        return results
```

**Multi-User Households:** The database supports multiple enrolled users. When multiple faces are detected simultaneously, each face is independently matched. The system tracks the "primary user" based on recency, proximity, and interaction frequency.

---

### 3.5 Emotion Detection

**Purpose:** Infer the user's current emotional state from their facial expression.

**Model:** DeepFace (with FER+ or AffectNet backend) or a custom lightweight CNN trained on AffectNet

**Why Emotion Detection?**
- Emotional state fundamentally determines what kind of assistant response is appropriate
- A user who appears stressed should not receive chatty, playful interactions
- Emotion trends over time can trigger wellness check-ins or reminders

**Detected Emotions:** angry, disgusted, fearful, happy, sad, surprised, neutral

```python
from deepface import DeepFace
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class EmotionResult:
    dominant_emotion: str
    emotion_scores: Dict[str, float]   # All emotions with confidence scores
    valence: str                        # "positive", "negative", "neutral"
    arousal: str                        # "high", "low" (energy level)

VALENCE_MAP = {
    "happy": "positive",
    "surprised": "positive",
    "neutral": "neutral",
    "sad": "negative",
    "angry": "negative",
    "fearful": "negative",
    "disgusted": "negative"
}

AROUSAL_MAP = {
    "angry": "high",
    "surprised": "high",
    "fearful": "high",
    "happy": "medium",
    "disgusted": "medium",
    "sad": "low",
    "neutral": "low"
}

class EmotionDetector:
    """
    Facial emotion recognition using DeepFace.
    Extracts dominant emotion, all scores, and high-level valence/arousal dimensions.
    """

    def __init__(self, backend: str = "opencv"):
        self.backend = backend
        # Options: "opencv", "ssd", "mtcnn", "retinaface"
        # "opencv" is fastest; "retinaface" is most accurate

    def detect_emotion(self, face_crop: np.ndarray) -> Optional[EmotionResult]:
        """
        Analyze emotion from a pre-cropped face region.
        Face crop should be extracted from the human detection bounding box.
        """
        try:
            results = DeepFace.analyze(
                img_path=face_crop,
                actions=["emotion"],
                detector_backend=self.backend,
                enforce_detection=False,
                silent=True
            )

            if not results:
                return None

            emotion_data = results[0]["emotion"]
            dominant = results[0]["dominant_emotion"]

            # Normalize scores to sum to 1.0
            total = sum(emotion_data.values())
            normalized_scores = {k: v / total for k, v in emotion_data.items()}

            return EmotionResult(
                dominant_emotion=dominant,
                emotion_scores=normalized_scores,
                valence=VALENCE_MAP.get(dominant, "neutral"),
                arousal=AROUSAL_MAP.get(dominant, "low")
            )

        except Exception:
            return None
```

**Temporal Smoothing:** Raw emotion outputs are noisy — a single frame may show "surprised" simply due to a blink. The memory system applies an exponential moving average over emotion scores across a rolling 3-second window to produce a stable emotional state estimate.

---

### 3.6 Gesture Recognition (MediaPipe)

**Purpose:** Detect and classify hand gestures and body poses to enable non-verbal interaction with the companion device.

**Model:** MediaPipe Hands + MediaPipe Pose

**Why MediaPipe?**
- Runs in real-time on CPU without GPU requirement
- Returns normalized 3D landmark coordinates (21 hand landmarks, 33 body landmarks)
- Designed for mobile/embedded deployment — ideal for edge devices
- Supports both hands and full-body pose estimation in a unified framework

**Supported Gestures:**
- Wave (greeting/farewell)
- Thumbs up (approval)
- Point (directing attention)
- Open palm / stop gesture
- Beckoning motion
- Body lean forward (engagement)
- Crossed arms (closed/disengaged)

```python
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class GestureResult:
    gesture_name: str          # Classified gesture label
    confidence: float
    hand_landmarks: Optional[List]   # Raw 21-point hand landmarks
    pose_landmarks: Optional[List]   # Raw 33-point body landmarks
    is_waving: bool
    body_orientation: str      # "facing", "turned_away", "side"

class GestureRecognizer:
    """
    MediaPipe-based hand gesture and body pose recognition system.
    Classifies high-level gestures from raw landmark data using heuristic rules
    and a lightweight classifier trained on common interactions.
    """

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,           # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Waving detection state
        self._wrist_history: List[Tuple[float, float]] = []
        self._wave_window = 15            # Number of frames to check for waving

    def _classify_hand_gesture(self, landmarks) -> Tuple[str, float]:
        """
        Rule-based gesture classification from 21 hand landmarks.
        Landmark indices follow MediaPipe convention:
        0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
        """
        if landmarks is None:
            return "none", 0.0

        lm = landmarks.landmark

        # Thumb tip vs thumb IP
        thumb_up = lm[4].y < lm[3].y < lm[2].y

        # Finger extended: tip above PIP joint
        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up = lm[16].y < lm[14].y
        pinky_up = lm[20].y < lm[18].y

        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

        if thumb_up and fingers_up == 0:
            return "thumbs_up", 0.90
        elif not thumb_up and fingers_up == 0:
            return "fist", 0.85
        elif fingers_up == 4 and not thumb_up:
            return "open_palm", 0.85
        elif index_up and not middle_up and not ring_up and not pinky_up:
            return "pointing", 0.80
        elif index_up and middle_up and not ring_up and not pinky_up:
            return "peace_sign", 0.80
        else:
            return "other", 0.50

    def _detect_waving(self, wrist_x: float) -> bool:
        """
        Detect waving by tracking lateral oscillation of wrist position.
        A wave is 2+ direction reversals within the tracking window.
        """
        self._wrist_history.append(wrist_x)
        if len(self._wrist_history) > self._wave_window:
            self._wrist_history.pop(0)

        if len(self._wrist_history) < 6:
            return False

        direction_changes = 0
        for i in range(2, len(self._wrist_history)):
            prev_dir = self._wrist_history[i-1] - self._wrist_history[i-2]
            curr_dir = self._wrist_history[i] - self._wrist_history[i-1]
            if prev_dir * curr_dir < 0 and abs(curr_dir) > 0.02:
                direction_changes += 1

        return direction_changes >= 2

    def _classify_body_orientation(self, pose_landmarks) -> str:
        if pose_landmarks is None:
            return "unknown"

        lm = pose_landmarks.landmark
        # Check visibility of shoulders
        left_shoulder_vis = lm[11].visibility
        right_shoulder_vis = lm[12].visibility

        if left_shoulder_vis > 0.7 and right_shoulder_vis > 0.7:
            # Both shoulders visible — facing camera
            shoulder_width = abs(lm[11].x - lm[12].x)
            return "facing" if shoulder_width > 0.15 else "leaning_away"
        elif left_shoulder_vis > 0.7:
            return "turned_right"
        elif right_shoulder_vis > 0.7:
            return "turned_left"
        else:
            return "turned_away"

    def process(self, rgb_frame: np.ndarray) -> GestureResult:
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)

        gesture_name = "none"
        confidence = 0.0
        is_waving = False
        hand_landmarks = None

        if hand_results.multi_hand_landmarks:
            primary_hand = hand_results.multi_hand_landmarks[0]
            hand_landmarks = primary_hand
            gesture_name, confidence = self._classify_hand_gesture(primary_hand)

            # Check for waving
            wrist = primary_hand.landmark[0]
            is_waving = self._detect_waving(wrist.x)
            if is_waving:
                gesture_name = "waving"
                confidence = 0.88

        pose_landmarks = pose_results.pose_landmarks if pose_results.pose_landmarks else None
        body_orientation = self._classify_body_orientation(pose_landmarks)

        return GestureResult(
            gesture_name=gesture_name,
            confidence=confidence,
            hand_landmarks=hand_landmarks,
            pose_landmarks=pose_landmarks,
            is_waving=is_waving,
            body_orientation=body_orientation
        )
```

---

### 3.7 Object Detection for Context

**Purpose:** Identify objects in the scene that provide contextual clues about the user's activities and environment.

**Model:** YOLOv8-small (same model reused from human detection, `classes` filtered to relevant objects)

**Why Reuse YOLOv8?** Rather than loading a separate model, the same YOLOv8 model used for human detection can simultaneously detect all 80 COCO classes. This avoids redundant inference passes and reduces memory footprint.

**Contextual Object Categories:**

| Category | Objects | Inferred Context |
|----------|---------|------------------|
| **Work** | laptop, keyboard, mouse, monitor, phone | Working / studying |
| **Meal** | cup, bottle, bowl, fork, knife | Eating / drinking |
| **Leisure** | book, remote, TV, gaming controller | Relaxing |
| **Exercise** | dumbbell, yoga mat, bicycle | Exercising |
| **Social** | multiple persons, chairs arranged together | Social gathering |

```python
from ultralytics import YOLO
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

# COCO class index → human-readable name (subset of interest)
CONTEXT_OBJECTS = {
    63: "laptop", 64: "mouse", 66: "keyboard", 67: "cell_phone",
    62: "tv", 72: "refrigerator", 73: "book", 74: "clock",
    39: "bottle", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 76: "scissors", 77: "teddy_bear"
}

ACTIVITY_INFERENCE = {
    frozenset(["laptop", "keyboard", "mouse"]): "working_at_computer",
    frozenset(["laptop"]): "using_laptop",
    frozenset(["book"]): "reading",
    frozenset(["cup", "bottle"]): "having_a_drink",
    frozenset(["bowl", "fork"]): "eating",
    frozenset(["cell_phone"]): "using_phone",
    frozenset(["tv"]): "watching_tv",
}

@dataclass
class ObjectDetectionResult:
    detected_objects: List[str]
    object_details: List[Dict]          # bbox, confidence, class per object
    inferred_activity: str              # High-level activity inference
    scene_complexity: str               # "sparse", "moderate", "cluttered"

class ContextObjectDetector:

    CONFIDENCE_THRESHOLD = 0.40

    def __init__(self, model_path: str = "yolov8s.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> ObjectDetectionResult:
        # Filter to context-relevant classes only
        target_classes = list(CONTEXT_OBJECTS.keys())
        results = self.model(frame, classes=target_classes,
                             conf=self.CONFIDENCE_THRESHOLD, verbose=False)

        detected_objects = []
        object_details = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                obj_name = CONTEXT_OBJECTS.get(class_id, "unknown")
                conf = float(box.conf[0])
                bbox = tuple(map(int, box.xyxy[0].tolist()))

                if obj_name not in detected_objects:
                    detected_objects.append(obj_name)

                object_details.append({
                    "name": obj_name,
                    "confidence": conf,
                    "bbox": bbox
                })

        # Infer activity from detected object combination
        detected_set = frozenset(detected_objects)
        inferred_activity = "idle"
        for obj_pattern, activity in ACTIVITY_INFERENCE.items():
            if obj_pattern.issubset(detected_set):
                inferred_activity = activity
                break

        # Scene complexity heuristic
        n = len(object_details)
        if n < 3:
            complexity = "sparse"
        elif n < 8:
            complexity = "moderate"
        else:
            complexity = "cluttered"

        return ObjectDetectionResult(
            detected_objects=detected_objects,
            object_details=object_details,
            inferred_activity=inferred_activity,
            scene_complexity=complexity
        )
```

---

### 3.8 Scene Understanding via Vision-Language Model (VLM)

**Purpose:** Generate a rich natural-language description of the scene that captures nuanced context not expressible through categorical detectors — environmental ambiance, activity nuance, social dynamics, and unusual events.

**Model:** Qwen-VL-Chat (7B), LLaVA-1.6-Mistral-7B, or MiniCPM-V-2 (for edge-optimized deployment)

**Why a VLM?**
- Object detectors and pose models operate on predefined categories. A VLM can describe novel situations, unusual configurations, and qualitative states that no list of classes can capture.
- Examples of VLM-only insights: "The user appears to be frustrated, with papers scattered on the desk", "The room is dimly lit and the user looks drowsy", "Two people appear to be having a discussion over something on the laptop screen"
- VLMs can answer targeted questions: "Is the user alone?", "What is the mood of the environment?"

**Deployment Strategy:** VLMs are expensive to run per-frame. This module runs **periodically** (every 5–10 seconds) or **event-driven** (when significant scene changes are detected).

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class SceneUnderstandingResult:
    scene_description: str         # Free-text scene description
    activity_summary: str          # What is the user doing?
    environment_notes: str         # Environmental context
    notable_events: Optional[str]  # Anything unusual or noteworthy

class VLMSceneAnalyzer:
    """
    Periodic scene understanding using a Vision-Language Model.
    Generates natural language descriptions and targeted Q&A about the scene.
    """

    SCENE_PROMPT = """You are a perceptive AI assistant analyzing a camera feed.
    Describe what you see concisely in 2-3 sentences.
    Focus on: (1) the person's activity and posture, (2) the objects in the scene,
    (3) the apparent mood or energy of the environment.
    Be factual and specific. Avoid speculation beyond what is visible.
    Format your response as JSON with keys: "scene_description", "activity_summary",
    "environment_notes", "notable_events"."""

    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat"):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def analyze_scene(self, frame: np.ndarray) -> SceneUnderstandingResult:
        image = Image.fromarray(frame)
        inputs = self.processor(
            text=self.SCENE_PROMPT,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0
            )

        response_text = self.processor.decode(output[0], skip_special_tokens=True)

        # Parse JSON from response
        import json, re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return SceneUnderstandingResult(
                    scene_description=data.get("scene_description", ""),
                    activity_summary=data.get("activity_summary", ""),
                    environment_notes=data.get("environment_notes", ""),
                    notable_events=data.get("notable_events")
                )
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text
        return SceneUnderstandingResult(
            scene_description=response_text,
            activity_summary="",
            environment_notes="",
            notable_events=None
        )
```

**Edge Optimization:** For resource-constrained devices, use **MiniCPM-V-2** (2B parameters, runs on 4GB RAM) or quantized LLaVA with **llama.cpp** (Q4 quantization). Alternatively, Qwen-VL can be run remotely via API if network is available and privacy policy permits.

---

## 4. Context and Memory System

### 4.1 Overview

Raw perception outputs from the pipeline are instantaneous snapshots. A single frame tells you "this person is smiling right now" — but it doesn't tell you whether they've been happy for the past 10 minutes, or whether they just arrived home. The Context and Memory System bridges this gap by maintaining temporal state.

The memory system operates at three time scales:

| Memory Type | Duration | Purpose |
|-------------|----------|---------|
| **Immediate Buffer** | Last 30 seconds | Smooth noisy detections, detect micro-events |
| **Session Memory** | Current session | Track activity trajectory, emotional arc |
| **Persistent Memory** | Days / weeks | Long-term habits, preferences, patterns |

### 4.2 Temporal Memory Buffer

```python
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

@dataclass
class PerceptionSnapshot:
    timestamp: float
    user_id: Optional[str]
    emotion: Optional[str]
    emotion_scores: Optional[Dict[str, float]]
    gesture: str
    detected_objects: List[str]
    activity: str
    body_orientation: str
    human_present: bool
    face_confidence: float

class TemporalMemoryBuffer:
    """
    Rolling time-windowed buffer of perception snapshots.
    Provides smoothed estimates and trend analysis over configurable windows.
    """

    def __init__(self, window_seconds: float = 30.0):
        self.window_seconds = window_seconds
        self.buffer: Deque[PerceptionSnapshot] = deque()

    def add(self, snapshot: PerceptionSnapshot):
        self.buffer.append(snapshot)
        self._prune_old()

    def _prune_old(self):
        cutoff = time.time() - self.window_seconds
        while self.buffer and self.buffer[0].timestamp < cutoff:
            self.buffer.popleft()

    def get_dominant_emotion(self) -> str:
        """Return the most frequent emotion over the buffer window."""
        emotions = [s.emotion for s in self.buffer if s.emotion]
        if not emotions:
            return "neutral"
        return max(set(emotions), key=emotions.count)

    def get_emotion_trend(self) -> str:
        """
        Detect emotional trend: improving, stable, or declining.
        Uses valence mapping to assign scores and compute linear trend.
        """
        VALENCE_SCORE = {"happy": 1, "neutral": 0, "sad": -1,
                          "angry": -2, "fearful": -1, "disgusted": -1, "surprised": 0.5}

        scored = [(s.timestamp, VALENCE_SCORE.get(s.emotion or "neutral", 0))
                  for s in self.buffer if s.emotion]

        if len(scored) < 5:
            return "stable"

        times = np.array([t for t, _ in scored])
        scores = np.array([v for _, v in scored])
        times_normalized = times - times[0]

        slope = np.polyfit(times_normalized, scores, 1)[0]

        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"

    def get_primary_activity(self) -> str:
        """Return the most common detected activity over the buffer window."""
        activities = [s.activity for s in self.buffer]
        if not activities:
            return "idle"
        return max(set(activities), key=activities.count)

    def get_presence_duration(self) -> float:
        """Return how long the user has been continuously present (in seconds)."""
        if not self.buffer:
            return 0.0

        # Walk back from most recent snapshot while user is present
        duration = 0.0
        snapshots = list(self.buffer)
        for i in range(len(snapshots) - 1, 0, -1):
            if snapshots[i].human_present:
                duration = snapshots[-1].timestamp - snapshots[i].timestamp
            else:
                break
        return duration

    def summarize(self) -> Dict:
        return {
            "dominant_emotion": self.get_dominant_emotion(),
            "emotion_trend": self.get_emotion_trend(),
            "primary_activity": self.get_primary_activity(),
            "presence_duration_seconds": self.get_presence_duration(),
            "buffer_size": len(self.buffer),
            "recent_gestures": list({s.gesture for s in list(self.buffer)[-10:]
                                      if s.gesture != "none"})
        }
```

### 4.3 Session Memory

Session memory persists for the duration of a single device session (from device wake until sleep or extended absence).

```python
@dataclass
class SessionMemory:
    session_id: str
    started_at: float
    user_id: Optional[str] = None
    activity_log: List[Dict] = field(default_factory=list)
    emotion_log: List[Dict] = field(default_factory=list)
    interaction_log: List[Dict] = field(default_factory=list)
    scene_descriptions: List[str] = field(default_factory=list)
    notable_events: List[str] = field(default_factory=list)

    def log_activity(self, activity: str, timestamp: float):
        if not self.activity_log or self.activity_log[-1]["activity"] != activity:
            self.activity_log.append({"activity": activity, "timestamp": timestamp})

    def log_emotion(self, emotion: str, timestamp: float):
        self.emotion_log.append({"emotion": emotion, "timestamp": timestamp})

    def log_interaction(self, interaction_type: str, content: str, timestamp: float):
        self.interaction_log.append({
            "type": interaction_type,
            "content": content,
            "timestamp": timestamp
        })

    def get_session_summary(self) -> str:
        duration = (time.time() - self.started_at) / 60
        activities = [e["activity"] for e in self.activity_log]
        dominant_activity = max(set(activities), key=activities.count) if activities else "idle"
        emotions = [e["emotion"] for e in self.emotion_log]
        dominant_emotion = max(set(emotions), key=emotions.count) if emotions else "neutral"

        return (f"Session duration: {duration:.1f} min. "
                f"Primary activity: {dominant_activity}. "
                f"Dominant mood: {dominant_emotion}. "
                f"Interactions: {len(self.interaction_log)}.")
```

### 4.4 Persistent Memory (Long-Term)

Long-term memory is stored as structured JSON in a local database (SQLite for simplicity, or a vector database for semantic search). It captures:

- Daily routine patterns (arrival time, departure time, typical activities per time-of-day)
- Emotional patterns (days/times when user typically appears stressed or happy)
- Interaction preferences (does the user respond positively to proactive conversation? music suggestions?)
- Named entity memory (names of people seen, frequently used objects)

```json
{
  "user_id": "user_001",
  "routine_patterns": {
    "monday": {
      "morning": "working_at_computer",
      "afternoon": "video_calls",
      "evening": "relaxing"
    }
  },
  "emotional_patterns": {
    "monday_morning": {"dominant": "neutral", "trend": "stable"},
    "friday_evening": {"dominant": "happy", "trend": "improving"}
  },
  "interaction_preferences": {
    "proactive_greetings": true,
    "music_suggestions": false,
    "reminder_frequency": "low",
    "conversation_style": "brief"
  }
}
```

---

## 5. Context Engine

### 5.1 Purpose

The Context Engine is the synthesis layer of the system. It takes raw outputs from the perception pipeline and the memory system and produces a single, structured **Context Object** — a machine-readable snapshot of the current situation that the AI Agent can reason about.

### 5.2 Context Object Schema

```python
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import json
import time

@dataclass
class ContextObject:
    # Identity
    user_id: Optional[str]
    user_display_name: Optional[str]
    identity_confidence: float

    # Emotional State
    emotion: str
    emotion_scores: Dict[str, float]
    emotion_trend: str                  # "improving", "stable", "declining"
    valence: str                        # "positive", "negative", "neutral"

    # Activity
    activity: str                       # Current high-level activity
    gesture: str                        # Most recent gesture
    body_orientation: str

    # Environment
    detected_objects: List[str]
    scene_description: str              # VLM-generated description
    environment_notes: str

    # Temporal
    presence_duration_seconds: float
    session_summary: str
    time_of_day: str                    # "morning", "afternoon", "evening", "night"

    # Events
    event: str                          # Primary event label
    notable_events: List[str]

    # Metadata
    timestamp: float
    context_version: str = "1.0"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_prompt_string(self) -> str:
        """
        Serialize context into a compact natural-language string
        suitable for inclusion in an LLM system prompt.
        """
        return (
            f"User: {self.user_display_name or 'unknown'} "
            f"(confidence: {self.identity_confidence:.0%}). "
            f"Emotion: {self.emotion} ({self.valence}, {self.emotion_trend} trend). "
            f"Activity: {self.activity}. "
            f"Gesture: {self.gesture}. "
            f"Objects in scene: {', '.join(self.detected_objects) or 'none'}. "
            f"Event: {self.event}. "
            f"Time of day: {self.time_of_day}. "
            f"Scene: {self.scene_description}"
        )
```

### 5.3 Context Builder

```python
import time
from typing import Optional

class ContextEngine:
    """
    Synthesizes perception outputs and memory summaries into a unified ContextObject.
    """

    TIME_BANDS = {
        (5, 12): "morning",
        (12, 17): "afternoon",
        (17, 21): "evening",
        (21, 24): "night",
        (0, 5): "night"
    }

    def _get_time_of_day(self) -> str:
        hour = time.localtime().tm_hour
        for (start, end), label in self.TIME_BANDS.items():
            if start <= hour < end:
                return label
        return "night"

    def _classify_event(self, face_result, gesture_result,
                         object_result, memory_summary: dict) -> str:
        """
        Classify the primary event based on combined perception signals.
        """
        if face_result and face_result[0].is_new_face:
            return "new_person_detected"
        if gesture_result.is_waving:
            return "user_waving"
        if gesture_result.gesture_name == "thumbs_up":
            return "user_approval_gesture"
        if memory_summary["primary_activity"] == "working_at_computer":
            return "user_working"
        if memory_summary["presence_duration_seconds"] > 3600:
            return "extended_session"
        if memory_summary["dominant_emotion"] in ["sad", "angry"]:
            return "user_appears_distressed"
        return "user_present"

    def build_context(
        self,
        face_results,
        emotion_result,
        gesture_result,
        object_result,
        scene_result,
        memory_summary: dict,
        session_memory
    ) -> ContextObject:

        # Extract identity
        user_id = None
        user_name = None
        id_confidence = 0.0
        if face_results:
            primary_face = face_results[0]
            user_id = primary_face.user_id
            user_name = primary_face.display_name
            id_confidence = primary_face.confidence

        # Extract emotion
        emotion = emotion_result.dominant_emotion if emotion_result else "neutral"
        emotion_scores = emotion_result.emotion_scores if emotion_result else {}
        valence = emotion_result.valence if emotion_result else "neutral"

        event = self._classify_event(face_results, gesture_result,
                                      object_result, memory_summary)

        return ContextObject(
            user_id=user_id,
            user_display_name=user_name,
            identity_confidence=id_confidence,
            emotion=emotion,
            emotion_scores=emotion_scores,
            emotion_trend=memory_summary.get("emotion_trend", "stable"),
            valence=valence,
            activity=memory_summary.get("primary_activity", object_result.inferred_activity),
            gesture=gesture_result.gesture_name,
            body_orientation=gesture_result.body_orientation,
            detected_objects=object_result.detected_objects,
            scene_description=scene_result.scene_description if scene_result else "",
            environment_notes=scene_result.environment_notes if scene_result else "",
            presence_duration_seconds=memory_summary.get("presence_duration_seconds", 0),
            session_summary=session_memory.get_session_summary(),
            time_of_day=self._get_time_of_day(),
            event=event,
            notable_events=memory_summary.get("notable_events", []),
            timestamp=time.time()
        )
```

---

## 6. Personalization Engine

### 6.1 Overview

The Personalization Engine learns from accumulated interaction history to adapt the device's behavior to each individual user. It is responsible for ensuring that the assistant feels personally tailored — not generic.

### 6.2 User Profile Structure

```python
@dataclass
class UserProfile:
    user_id: str
    display_name: str
    enrolled_at: float

    # Interaction preferences (learned over time)
    preferred_interaction_style: str    # "brief", "conversational", "proactive", "reactive"
    sensitivity_to_emotion: bool        # Does user appreciate emotional comments?
    reminder_tolerance: str             # "high", "medium", "low"
    humor_preference: bool
    preferred_topics: List[str]         # ["technology", "music", "health"]

    # Behavioral patterns (computed from session history)
    typical_arrival_time: Optional[str]
    typical_departure_time: Optional[str]
    primary_use_location: str           # "desk", "couch", "kitchen"
    average_session_duration_minutes: float
    most_common_emotion: str
    most_common_activity: str

    # Feedback history (explicit and implicit)
    positive_interactions: int          # Interactions user responded positively to
    negative_interactions: int          # Interactions user dismissed or ignored
    last_interaction: Optional[float]
```

### 6.3 Preference Learning

The engine learns through two signals:

**Explicit feedback:** User responds positively (smiles, engages), negatively (dismisses, ignores, shows frustration), or provides direct rating.

**Implicit feedback:** Inferred from emotional state and behavioral response after an assistant action.

```python
class PersonalizationEngine:

    FEEDBACK_DECAY = 0.95    # Weight of historical preference vs recent interactions

    def update_preference(self, user_id: str, action_type: str,
                          outcome: str, context: ContextObject):
        """
        Update preference weights based on interaction outcome.
        outcome: "positive", "neutral", "negative"
        """
        profile = self.load_profile(user_id)
        weight = {"positive": +1, "neutral": 0, "negative": -1}[outcome]

        if action_type not in profile.action_weights:
            profile.action_weights[action_type] = 0.5   # Start neutral

        current = profile.action_weights[action_type]
        profile.action_weights[action_type] = (
            current * self.FEEDBACK_DECAY + (0.5 + weight * 0.1)
        )
        self.save_profile(profile)

    def get_behavior_instructions(self, profile: UserProfile,
                                   context: ContextObject) -> str:
        """
        Generate natural language behavioral instructions for the agent
        based on the user's learned profile.
        """
        instructions = []

        if profile.preferred_interaction_style == "brief":
            instructions.append("Keep responses concise and to the point.")
        elif profile.preferred_interaction_style == "conversational":
            instructions.append("Engage warmly and conversationally.")

        if not profile.sensitivity_to_emotion:
            instructions.append("Avoid commenting on emotional state unless asked.")

        if profile.reminder_tolerance == "low":
            instructions.append("Do not send reminders unless urgently requested.")

        if context.emotion in ["sad", "angry"] and profile.sensitivity_to_emotion:
            instructions.append("The user appears to be having a difficult time. Be gentle and supportive.")

        return " ".join(instructions)
```

---

## 7. AI Agent Layer

### 7.1 Overview

The AI Agent (named **Aura**) is the intelligence core of the system. It receives the structured context object and behavioral instructions from the personalization engine, and decides what action to take — or whether to take no action at all.

Aura is implemented as an LLM-based reasoning agent. The context object and behavioral instructions are serialized into the system prompt. The agent reasons about the situation and produces a structured action decision.

### 7.2 Agent Architecture

```python
import json
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class AgentAction:
    action_type: str           # "greet", "comment", "reminder", "reaction", "silence", "question"
    message: Optional[str]     # Text to speak/display
    tone: str                  # "warm", "playful", "concerned", "professional", "excited"
    urgency: str               # "immediate", "delayed_5s", "next_opportunity"
    animation: Optional[str]   # Device animation to play
    reasoning: str             # Agent's internal reasoning (for logging/debug)

class AuraAgent:
    """
    LLM-based AI agent that interprets context and decides device actions.
    Uses a structured prompt template to produce consistent, parseable outputs.
    """

    SYSTEM_PROMPT_TEMPLATE = """
You are Aura, an empathetic AI companion device with a warm personality.
You observe a user in their environment and decide how to respond.

Your personality:
- Observant and perceptive, but not intrusive
- Warm and supportive, but not over-eager
- Occasionally playful, but always appropriate
- Prioritizes the user's comfort and wellbeing

Current context:
{context_string}

User behavioral profile:
{behavior_instructions}

Available actions:
- greet: Acknowledge the user's presence warmly
- comment: Observe something about the user's activity or environment
- reminder: Provide a helpful reminder based on time or activity duration
- reaction: React to a detected gesture
- question: Ask a brief, relevant question
- silence: Take no action (often the right choice)

DECISION RULES:
1. If the user just arrived, greet them
2. If the user is in deep focus (working 30+ min), do not interrupt
3. If the user looks distressed, offer gentle support
4. If a meaningful gesture was detected, react appropriately
5. If the user has been at desk 90+ min, suggest a break
6. If nothing significant is happening, choose silence
7. Never repeat the same interaction type within 5 minutes

Respond with a JSON object:
{{
  "action_type": "...",
  "message": "...",
  "tone": "...",
  "urgency": "...",
  "animation": "...",
  "reasoning": "..."
}}
"""

    def __init__(self, llm_client):
        self.llm = llm_client           # Can be local LLM (Ollama) or API (Anthropic/OpenAI)
        self.last_action_log: List[AgentAction] = []
        self.action_cooldowns: dict = {}

    def _apply_cooldown_check(self, action_type: str) -> bool:
        """Return True if action is allowed (not in cooldown)."""
        import time
        cooldown_seconds = {
            "greet": 300,
            "reminder": 600,
            "comment": 180,
            "question": 300,
            "reaction": 30,
            "silence": 0
        }.get(action_type, 120)

        last_time = self.action_cooldowns.get(action_type, 0)
        return (time.time() - last_time) >= cooldown_seconds

    def decide(self, context: "ContextObject",
               behavior_instructions: str) -> AgentAction:
        """
        Run the agent reasoning loop and produce an action decision.
        """
        import time

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            context_string=context.to_prompt_string(),
            behavior_instructions=behavior_instructions
        )

        response = self.llm.complete(system_prompt)

        try:
            data = json.loads(response)
            action = AgentAction(**data)
        except (json.JSONDecodeError, TypeError):
            # Fallback to silence on parse failure
            action = AgentAction(
                action_type="silence",
                message=None,
                tone="neutral",
                urgency="next_opportunity",
                animation=None,
                reasoning="Parse failure — defaulting to silence"
            )

        # Apply cooldown enforcement
        if not self._apply_cooldown_check(action.action_type):
            action = AgentAction(
                action_type="silence",
                message=None,
                tone="neutral",
                urgency="next_opportunity",
                animation=None,
                reasoning=f"Action '{action.action_type}' in cooldown — choosing silence"
            )
        else:
            self.action_cooldowns[action.action_type] = time.time()

        self.last_action_log.append(action)
        return action
```

### 7.3 LLM Backend Options

| Backend | Model | Deployment | Latency | Cost |
|---------|-------|------------|---------|------|
| **Ollama (local)** | Llama-3.1-8B-Instruct | Edge (GPU) | ~0.5–2s | Free |
| **Anthropic API** | Claude Haiku | Cloud | ~0.3–0.8s | Per-token |
| **OpenAI API** | GPT-4o-mini | Cloud | ~0.3–1s | Per-token |
| **LM Studio** | Mistral-7B-Instruct | Edge (CPU) | ~2–5s | Free |

For privacy-first deployments, Ollama with a quantized local model is preferred.

---

## 8. Assistant Behavior Layer

### 8.1 Overview

The Behavior Layer translates agent action decisions into real-world outputs. It manages text-to-speech, display rendering, LED animations, and sound playback.

### 8.2 Behavior Modules

#### 8.2.1 Greeting Module

Triggered when: user arrives, user returns after absence, new user detected.

```python
class GreetingBehavior:

    GREETING_TEMPLATES = {
        "morning": [
            "Good morning, {name}! Ready for the day?",
            "Morning! I hope you're feeling energized today.",
        ],
        "afternoon": [
            "Hey {name}, welcome back!",
            "Good afternoon! How's your day going?",
        ],
        "evening": [
            "Good evening, {name}. How was your day?",
            "Evening! Winding down for the night?",
        ]
    }

    def generate_greeting(self, user_name: str, time_of_day: str,
                           emotion: str, absence_duration: float) -> str:
        templates = self.GREETING_TEMPLATES.get(time_of_day,
                                                 self.GREETING_TEMPLATES["afternoon"])
        import random
        base = random.choice(templates).format(name=user_name or "there")

        # Add contextual enrichment
        if absence_duration > 3600:
            hours = int(absence_duration / 3600)
            base += f" You've been away for about {hours} hour{'s' if hours > 1 else ''}."

        if emotion == "happy":
            base += " You look like you're in a great mood!"
        elif emotion in ["sad", "angry"]:
            base += " Is everything alright?"

        return base
```

#### 8.2.2 Gesture Reaction Module

Triggered when: significant gesture detected.

```python
GESTURE_RESPONSES = {
    "waving": {
        "message": "Hey! 👋",
        "animation": "wave_back",
        "tone": "playful"
    },
    "thumbs_up": {
        "message": "Awesome! Glad things are going well!",
        "animation": "happy_bounce",
        "tone": "excited"
    },
    "open_palm": {
        "message": "Sure, I'll hold off for now.",
        "animation": "calm_nod",
        "tone": "professional"
    },
    "pointing": {
        "message": "I see you're pointing at something. Need help with that?",
        "animation": "look_direction",
        "tone": "curious"
    }
}
```

#### 8.2.3 Reminder Module

Triggered when: extended focus session, time-of-day pattern, scheduled events.

```python
class ReminderBehavior:

    REMINDER_RULES = [
        {
            "condition": lambda ctx: ctx.presence_duration_seconds >= 5400,  # 90 min
            "message": "You've been at your desk for a while. Time for a quick stretch?",
            "type": "break_reminder"
        },
        {
            "condition": lambda ctx: ctx.time_of_day == "morning" and ctx.activity == "working_at_computer",
            "message": "Don't forget to eat breakfast if you haven't!",
            "type": "meal_reminder"
        }
    ]

    def check_reminders(self, context: "ContextObject") -> Optional[str]:
        for rule in self.REMINDER_RULES:
            if rule["condition"](context):
                return rule["message"]
        return None
```

#### 8.2.4 Emotional Support Module

Triggered when: user displays sustained negative emotion.

```python
EMOTIONAL_SUPPORT_RESPONSES = {
    "sad": [
        "You seem a bit down. I'm here if you need to talk.",
        "Sometimes things are tough. You've got this.",
    ],
    "angry": [
        "Looks like something's frustrating you. Deep breath?",
        "I can sense some tension. Need a moment?",
    ],
    "fearful": [
        "You seem a bit anxious. Want me to put on something calming?",
    ]
}
```

### 8.3 Output Modalities

| Modality | Technology | Purpose |
|----------|-----------|---------|
| **Voice** | TTS (Coqui / pyttsx3 / cloud TTS) | Primary verbal communication |
| **Display** | Small OLED or 4" LCD | Show text, faces, status icons |
| **LED Ring** | NeoPixel / WS2812 | Emotional state indicator (color) |
| **Sound FX** | pygame / aplay | Non-verbal audio cues |
| **Servo/motor** | Via microcontroller | Head turn, nod gestures |

```python
class BehaviorExecutor:
    """
    Executes agent actions through appropriate output modalities.
    """

    def execute(self, action: AgentAction, tts_engine, display, led_controller):
        if action.action_type == "silence":
            return

        # Text-to-speech output
        if action.message:
            tts_engine.speak(action.message, tone=action.tone)

        # LED emotional indicator
        led_colors = {
            "warm": (255, 150, 50),
            "playful": (100, 200, 255),
            "concerned": (255, 100, 100),
            "excited": (100, 255, 100),
            "professional": (200, 200, 255)
        }
        led_controller.set_color(*led_colors.get(action.tone, (255, 255, 255)))

        # Display update
        if action.message:
            display.show_message(action.message)

        # Animation trigger
        if action.animation:
            display.play_animation(action.animation)
```

---

## 9. System Performance Strategy

### 9.1 The Core Challenge

Running YOLOv8, InsightFace, MediaPipe, DeepFace, and a VLM simultaneously on edge hardware is computationally expensive. Without careful design, the pipeline would either drop frames, incur high latency, or exhaust memory. The performance strategy decouples slow operations from fast operations using asynchronous, tiered processing.

### 9.2 Tiered Processing Architecture

```
TIER 1 — REALTIME (every frame, ~30 FPS target)
────────────────────────────────────────────────────────────────────
  YOLOv8-nano: Human presence check         ~5ms  (GPU) / ~20ms (CPU)
  MediaPipe: Gesture detection              ~8ms  (CPU)
  Frame quality check                       ~1ms
  ────────────────────────────────────────────────────────────────────
  Total target: < 33ms per frame

TIER 2 — PERIODIC (every 2–5 seconds)
────────────────────────────────────────────────────────────────────
  InsightFace: Face recognition             ~50ms (GPU) / ~200ms (CPU)
  DeepFace: Emotion detection               ~30ms (GPU) / ~100ms (CPU)
  Object detection (YOLOv8-small)           ~15ms (GPU) / ~60ms (CPU)
  Memory buffer update                      ~2ms
  ────────────────────────────────────────────────────────────────────
  Total target: < 300ms (async, non-blocking)

TIER 3 — EVENT-DRIVEN (on state change or every 10 seconds)
────────────────────────────────────────────────────────────────────
  VLM Scene Analysis (Qwen-VL / LLaVA)     ~500ms – 3s
  Context Engine build                      ~5ms
  Personalization Engine enrichment         ~5ms
  Agent LLM reasoning (Aura)               ~200ms – 1s
  ────────────────────────────────────────────────────────────────────
  Total target: < 5s (fully async, background thread)
```

### 9.3 Asynchronous Pipeline Implementation

```python
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

class PerceptionOrchestrator:
    """
    Manages the tiered, asynchronous execution of the perception pipeline.
    Ensures realtime components run every frame while expensive components
    run on their own schedule without blocking the main loop.
    """

    def __init__(self, tier1_components, tier2_components, tier3_components):
        self.tier1 = tier1_components
        self.tier2 = tier2_components
        self.tier3 = tier3_components

        self.executor = ThreadPoolExecutor(max_workers=4)

        self.tier2_interval = 3.0       # seconds
        self.tier3_interval = 10.0      # seconds

        self._last_tier2 = 0
        self._last_tier3 = 0

        # Shared state — updated by background threads, read by main loop
        self._latest_face_result = None
        self._latest_emotion_result = None
        self._latest_object_result = None
        self._latest_scene_result = None
        self._latest_context = None
        self._lock = threading.Lock()

    def run_tier1(self, frame):
        """Runs synchronously every frame."""
        human_detections = self.tier1["detector"].detect(frame)
        gesture = self.tier1["gesture"].process(frame)
        return human_detections, gesture

    def schedule_tier2(self, frame):
        """Schedules tier 2 in background thread pool."""
        if time.time() - self._last_tier2 >= self.tier2_interval:
            self._last_tier2 = time.time()
            self.executor.submit(self._run_tier2, frame.copy())

    def _run_tier2(self, frame):
        faces = self.tier2["face_recognition"].recognize(frame)
        if faces:
            emotion = self.tier2["emotion"].detect_emotion(
                frame[faces[0].face_bbox[1]:faces[0].face_bbox[3],
                      faces[0].face_bbox[0]:faces[0].face_bbox[2]]
            )
        else:
            emotion = None
        objects = self.tier2["objects"].detect(frame)

        with self._lock:
            self._latest_face_result = faces
            self._latest_emotion_result = emotion
            self._latest_object_result = objects

    def schedule_tier3(self, frame):
        """Schedules tier 3 in background thread pool."""
        if time.time() - self._last_tier3 >= self.tier3_interval:
            self._last_tier3 = time.time()
            self.executor.submit(self._run_tier3, frame.copy())

    def _run_tier3(self, frame):
        scene = self.tier3["vlm"].analyze_scene(frame)
        with self._lock:
            self._latest_scene_result = scene
        # Trigger context build and agent decision
        self.tier3["context_engine_callback"]()

    def get_latest_results(self):
        with self._lock:
            return {
                "face": self._latest_face_result,
                "emotion": self._latest_emotion_result,
                "objects": self._latest_object_result,
                "scene": self._latest_scene_result
            }
```

### 9.4 Model Optimization Techniques

| Technique | Method | Speedup |
|-----------|--------|---------|
| **FP16 inference** | `model.half()` on CUDA | 1.5–2× |
| **TensorRT export** | YOLOv8 → `.engine` | 2–4× on Jetson |
| **ONNX Runtime** | General models | 1.3–2× |
| **INT8 quantization** | Post-training quantization | 2–4× (slight accuracy drop) |
| **Frame skipping** | Tier 1 every N frames | Linear |
| **ROI cropping** | Process only person bounding box | 2–5× for face/emotion models |
| **Model caching** | Keep models in GPU memory | Eliminates load latency |

---

## 10. Hardware Architecture

### 10.1 System Components

```
┌────────────────────────────────────────────────────────────┐
│                    COMPANION DEVICE UNIT                   │
│                                                            │
│  ┌───────────────┐    ┌───────────────────────────────┐   │
│  │   USB/CSI     │    │       Edge Compute Module     │   │
│  │   Camera      │───▶│   (Laptop / Jetson / NUC)     │   │
│  │  (1080p 30fps)│    │                               │   │
│  └───────────────┘    │  - YOLOv8, MediaPipe          │   │
│                       │  - InsightFace, DeepFace       │   │
│  ┌───────────────┐    │  - VLM (Qwen-VL / LLaVA)      │   │
│  │  Microphone   │───▶│  - Agent LLM (Ollama)          │   │
│  │  (Optional)   │    │  - Memory DB (SQLite)          │   │
│  └───────────────┘    └───────────────┬───────────────┘   │
│                                       │ USB Serial / GPIO  │
│  ┌───────────────────────────────────▼───────────────┐    │
│  │              Microcontroller (ESP32 / Arduino)     │    │
│  │   - Servo control (head movement)                  │    │
│  │   - NeoPixel LED ring control                      │    │
│  │   - Speaker amp (I2S DAC)                          │    │
│  │   - OLED display (SSD1306 / SSD1351)               │    │
│  │   - Touch sensor input                             │    │
│  └───────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────┘
```

### 10.2 Hardware Configuration Options

#### Option A: Developer / High-Performance

| Component | Specification |
|-----------|--------------|
| Camera | Logitech BRIO 4K (USB 3.0) |
| Compute | NVIDIA Jetson Orin NX 16GB |
| Storage | 128GB NVMe SSD |
| Microcontroller | ESP32-S3 |
| Display | 4" IPS TFT (480×320) |
| Speaker | 3W I2S amplifier + small speaker |
| LEDs | 24-LED WS2812B ring |
| Power | 12V/5A DC or Li-Po battery pack |

#### Option B: Portable / Embedded

| Component | Specification |
|-----------|--------------|
| Camera | IMX219 CSI module (8MP) |
| Compute | Raspberry Pi 5 (8GB) + Hailo-8 AI accelerator |
| Storage | 64GB microSD + USB3 SSD |
| Microcontroller | Arduino Nano 33 IoT |
| Display | 2.4" OLED (SSD1351) |
| Speaker | 1W I2S audio bonnet |
| LEDs | 12-LED NeoPixel ring |

#### Option C: Laptop-Based (Development / Prototype)

| Component | Specification |
|-----------|--------------|
| Camera | USB webcam (Logitech C920) |
| Compute | MacBook Pro M3 / Intel i7 laptop |
| Storage | Onboard SSD |
| Output | Laptop display + system speakers |
| LED/servo | Optional USB-connected Arduino |

### 10.3 Microcontroller Communication Protocol

```python
import serial
import json
import threading

class MicrocontrollerBridge:
    """
    Serial communication bridge to ESP32/Arduino microcontroller.
    Sends JSON command objects over USB serial.
    """

    COMMANDS = {
        "led_color": {"r": 0, "g": 0, "b": 0},
        "play_animation": {"name": ""},
        "servo_move": {"axis": "head_yaw", "degrees": 0},
        "display_text": {"text": "", "color": "white"}
    }

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        self._lock = threading.Lock()

    def send_command(self, command_type: str, params: dict):
        payload = json.dumps({"cmd": command_type, **params}) + "\n"
        with self._lock:
            self.ser.write(payload.encode())

    def set_led_color(self, r: int, g: int, b: int):
        self.send_command("led_color", {"r": r, "g": g, "b": b})

    def play_animation(self, name: str):
        self.send_command("play_animation", {"name": name})

    def move_head(self, yaw_degrees: float, pitch_degrees: float = 0):
        self.send_command("servo_move", {"yaw": yaw_degrees, "pitch": pitch_degrees})
```

---

## 11. Data Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                       FULL SYSTEM DATA FLOW DIAGRAM                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌──────────────┐
  │  USB / CSI   │
  │    Camera    │  Raw BGR frames @ 30FPS
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────┐
  │   Frame Capture Thread       │  Thread-safe Queue (bounded, size=2)
  │   (CameraInputLayer)         │  Always serves freshest frame
  └──────┬───────────────────────┘
         │
         ▼
  ┌──────────────────────────────┐
  │   Frame Processor            │  Resize → CLAHE → Quality Check
  │   (OpenCV)                   │  Outputs: BGR frame, RGB frame, metadata
  └──────┬───────────────────────┘
         │
         ├─────────────────────────────────────────────────────┐
         │   TIER 1 (Synchronous, every frame)                 │
         │                                                     │
         ▼                                                     ▼
  ┌──────────────────┐                               ┌──────────────────┐
  │  YOLOv8-nano     │                               │  MediaPipe       │
  │  Human Detector  │                               │  Gesture + Pose  │
  │                  │                               │  Recognizer      │
  │  Output:         │                               │                  │
  │  - BBox list     │                               │  Output:         │
  │  - Confidence    │                               │  - Gesture name  │
  │  - Position      │                               │  - Body orient.  │
  └────────┬─────────┘                               └────────┬─────────┘
           │                                                  │
           └──────────────────┬───────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────┐
  │        PERCEPTION ORCHESTRATOR                   │
  │  (Manages tiered scheduling + shared state)      │
  └──────────────────┬───────────────────────────────┘
                     │
         ┌───────────┴─────────────────┐
         │   TIER 2 (Async, every 3s)  │
         │                             │
         ▼                             ▼                      ▼
  ┌──────────────┐            ┌────────────────┐    ┌──────────────────┐
  │ InsightFace  │            │  DeepFace      │    │  YOLOv8-small    │
  │ ArcFace      │            │  Emotion CNN   │    │  Object Detect.  │
  │              │            │                │    │                  │
  │ Output:      │            │  Output:       │    │  Output:         │
  │ - user_id    │            │  - emotion     │    │  - object list   │
  │ - confidence │            │  - scores      │    │  - activity      │
  └──────┬───────┘            └───────┬────────┘    └────────┬─────────┘
         │                            │                      │
         └────────────────────────────┴──────────────────────┘
                                      │
                                      ▼
  ┌──────────────────────────────────────────────────────┐
  │             TEMPORAL MEMORY BUFFER                   │
  │   Rolling 30s window of PerceptionSnapshots          │
  │   Computes: dominant emotion, trend, activity,       │
  │             presence duration, recent gestures       │
  └──────────────────────────┬───────────────────────────┘
                             │
              ┌──────────────┴────────────────────────────────┐
              │   TIER 3 (Async, every 10s / on event)        │
              │                                               │
              ▼                                               │
  ┌──────────────────────┐                                   │
  │  Vision-Language     │   Scene description               │
  │  Model (Qwen-VL /    │──────────────────────────────────▶│
  │  LLaVA / MiniCPM-V) │                                    │
  └──────────────────────┘                                    │
                                                              ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                      CONTEXT ENGINE                            │
  │  Combines: perception outputs + memory summary + VLM output    │
  │  Produces: structured ContextObject (JSON)                     │
  └──────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                   PERSONALIZATION ENGINE                     │
  │  Enriches context with user profile + behavior instructions  │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                      AURA AI AGENT                           │
  │  LLM reasoning over context → AgentAction decision           │
  │  Applies cooldown rules → Final action selection             │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
  ┌─────────────┐       ┌──────────────┐       ┌─────────────────┐
  │  TTS Engine │       │ OLED Display │       │  Microcontroller│
  │  (Voice)    │       │ (Text/Anim.) │       │  - LED ring     │
  │             │       │              │       │  - Servo head   │
  │  Coqui TTS  │       │  pygame /    │       │  - Sound FX     │
  │  pyttsx3    │       │  tkinter     │       │  (ESP32/Arduino)│
  └─────────────┘       └──────────────┘       └─────────────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   USER / ENVIRONMENT   │
                    └────────────────────────┘
```

---

## 12. Example Context Outputs

### 12.1 Morning Arrival — Happy User

```json
{
  "user_id": "user_001",
  "user_display_name": "Alex",
  "identity_confidence": 0.91,
  "emotion": "happy",
  "emotion_scores": {
    "happy": 0.72,
    "neutral": 0.20,
    "surprised": 0.05,
    "sad": 0.02,
    "angry": 0.01
  },
  "emotion_trend": "stable",
  "valence": "positive",
  "activity": "just_arrived",
  "gesture": "waving",
  "body_orientation": "facing",
  "detected_objects": ["backpack", "coffee_cup"],
  "scene_description": "A person has just entered the room carrying a backpack and a coffee cup. They appear energized and are smiling at the camera.",
  "environment_notes": "Morning light from window. Room is tidy.",
  "presence_duration_seconds": 12.0,
  "session_summary": "Session started 12 seconds ago. No prior activity.",
  "time_of_day": "morning",
  "event": "user_waving",
  "notable_events": [],
  "timestamp": 1741257600.0,
  "context_version": "1.0"
}
```

**Expected Agent Action:**
```json
{
  "action_type": "greet",
  "message": "Good morning, Alex! You're looking great today. Starting with coffee — smart move!",
  "tone": "warm",
  "urgency": "immediate",
  "animation": "wave_back",
  "reasoning": "User just arrived, is waving, appears happy, holding coffee. Morning greeting with light observation is appropriate."
}
```

---

### 12.2 Deep Focus Work Session

```json
{
  "user_id": "user_001",
  "user_display_name": "Alex",
  "identity_confidence": 0.87,
  "emotion": "neutral",
  "emotion_scores": {
    "neutral": 0.65,
    "happy": 0.10,
    "sad": 0.05,
    "angry": 0.15,
    "surprised": 0.05
  },
  "emotion_trend": "stable",
  "valence": "neutral",
  "activity": "working_at_computer",
  "gesture": "none",
  "body_orientation": "facing",
  "detected_objects": ["laptop", "keyboard", "mouse", "cup", "monitor"],
  "scene_description": "Person is seated at a desk, intensely focused on a laptop screen. Multiple documents appear open. A cup of coffee is nearby. The environment is quiet and task-oriented.",
  "environment_notes": "Artificial overhead lighting. Desk is organized.",
  "presence_duration_seconds": 6300.0,
  "session_summary": "Session duration: 105 min. Primary activity: working_at_computer. Dominant mood: neutral. Interactions: 1.",
  "time_of_day": "afternoon",
  "event": "extended_session",
  "notable_events": ["no_breaks_detected_90min"],
  "timestamp": 1741272000.0,
  "context_version": "1.0"
}
```

**Expected Agent Action:**
```json
{
  "action_type": "reminder",
  "message": "Hey Alex, you've been working for over an hour and a half. Maybe stretch your legs for a few minutes?",
  "tone": "warm",
  "urgency": "next_opportunity",
  "animation": "gentle_pulse",
  "reasoning": "User has been at desk 105 min with no breaks detected. Break reminder is appropriate. Tone should be gentle, not disruptive."
}
```

---

### 12.3 Distressed User

```json
{
  "user_id": "user_001",
  "user_display_name": "Alex",
  "identity_confidence": 0.83,
  "emotion": "sad",
  "emotion_scores": {
    "sad": 0.55,
    "neutral": 0.25,
    "angry": 0.12,
    "fearful": 0.05,
    "happy": 0.03
  },
  "emotion_trend": "declining",
  "valence": "negative",
  "activity": "idle",
  "gesture": "none",
  "body_orientation": "facing",
  "detected_objects": ["phone", "tissues"],
  "scene_description": "Person is sitting still, looking downward. They appear emotionally withdrawn. A phone is nearby face-down, and a box of tissues is visible on the desk.",
  "environment_notes": "Dim lighting. Room is quiet.",
  "presence_duration_seconds": 1800.0,
  "session_summary": "Session duration: 30 min. Primary activity: idle. Dominant mood: sad, declining trend. Interactions: 0.",
  "time_of_day": "evening",
  "event": "user_appears_distressed",
  "notable_events": ["sustained_negative_emotion_15min"],
  "timestamp": 1741291200.0,
  "context_version": "1.0"
}
```

**Expected Agent Action:**
```json
{
  "action_type": "comment",
  "message": "Hey Alex... I notice you seem a bit down tonight. I'm here if you want to talk, or I can just keep you company quietly.",
  "tone": "concerned",
  "urgency": "delayed_5s",
  "animation": "soft_glow_purple",
  "reasoning": "User has shown sustained sadness with declining trend for 15+ minutes. Gentle, non-intrusive supportive comment is appropriate. Offer company without pressure."
}
```

---

### 12.4 Unknown Person Detected

```json
{
  "user_id": null,
  "user_display_name": null,
  "identity_confidence": 0.0,
  "emotion": "neutral",
  "emotion_scores": {
    "neutral": 0.80,
    "happy": 0.12,
    "surprised": 0.08
  },
  "emotion_trend": "stable",
  "valence": "neutral",
  "activity": "idle",
  "gesture": "none",
  "body_orientation": "facing",
  "detected_objects": ["bag"],
  "scene_description": "An unrecognized individual is standing in the room, appearing to look around curiously.",
  "environment_notes": "Normal lighting.",
  "presence_duration_seconds": 5.0,
  "session_summary": "Session started 5 seconds ago.",
  "time_of_day": "afternoon",
  "event": "new_person_detected",
  "notable_events": ["unrecognized_face"],
  "timestamp": 1741276800.0,
  "context_version": "1.0"
}
```

**Expected Agent Action:**
```json
{
  "action_type": "greet",
  "message": "Hello! I don't think we've met before. I'm Aura. Would you like me to remember you?",
  "tone": "warm",
  "urgency": "immediate",
  "animation": "curious_tilt",
  "reasoning": "Unrecognized face detected. Friendly introduction and enrollment offer is appropriate."
}
```

---

### 12.5 User Leaving — Farewell

```json
{
  "user_id": "user_001",
  "user_display_name": "Alex",
  "identity_confidence": 0.85,
  "emotion": "neutral",
  "emotion_scores": {
    "neutral": 0.60,
    "happy": 0.25,
    "sad": 0.10,
    "surprised": 0.05
  },
  "emotion_trend": "stable",
  "valence": "neutral",
  "activity": "packing_up",
  "gesture": "waving",
  "body_orientation": "turned_away",
  "detected_objects": ["laptop", "backpack", "jacket"],
  "scene_description": "Person is gathering their belongings, picking up a backpack and jacket. They appear to be preparing to leave and glanced back toward the camera.",
  "environment_notes": "Lights being turned off in background.",
  "presence_duration_seconds": 28800.0,
  "session_summary": "Session duration: 480 min. Primary activity: working_at_computer. Dominant mood: neutral. Interactions: 6.",
  "time_of_day": "evening",
  "event": "user_waving",
  "notable_events": ["long_session_8hr"],
  "timestamp": 1741305600.0,
  "context_version": "1.0"
}
```

**Expected Agent Action:**
```json
{
  "action_type": "greet",
  "message": "Heading out, Alex? You put in a solid 8 hours today. Make sure to rest up tonight. See you tomorrow!",
  "tone": "warm",
  "urgency": "immediate",
  "animation": "wave_goodbye",
  "reasoning": "User is leaving, waving, carrying bag and jacket. Long session noted. Warm farewell with positive reinforcement is appropriate."
}
```

---

## 13. Future Improvements

### 13.1 Voice Interaction and Multimodal I/O

The current architecture is primarily vision-driven. Adding a dedicated microphone and speaker system would enable full bidirectional voice interaction.

**Planned additions:**

- **Wake word detection:** Lightweight on-device model (Porcupine, Snowboy) listening for "Hey Aura"
- **Automatic Speech Recognition (ASR):** Whisper.cpp (local) or Deepgram API for transcription
- **Text-to-Speech (TTS):** Coqui TTS with custom voice training to give Aura a distinctive, consistent voice
- **Voice activity detection (VAD):** Silero VAD to gate microphone recording efficiently
- **Multimodal agent input:** Agent receives both vision context AND transcribed speech in each prompt

This would enable natural two-way conversations: "Aura, what time is my meeting?" combined with visual awareness of whether the user is free to receive the answer now.

### 13.2 Long-Term Episodic Memory

Current persistent memory captures statistical patterns. True episodic memory would allow Aura to recall specific past events:

- "Last Tuesday you seemed stressed around 3pm — is today going better?"
- "You mentioned wanting to call your mom. It's been a week since she's been mentioned."
- "You've been working late every day this week. That's unusual for you."

**Implementation approach:**
- Embed session summaries as vector embeddings (using `sentence-transformers`)
- Store in a local vector database (Chroma, FAISS, or LanceDB)
- Retrieve relevant memories via semantic similarity search at context-build time
- Include retrieved memories in agent system prompt as "memory context"

### 13.3 Activity Recognition with Video Understanding

Current activity recognition is based on object co-occurrence heuristics. A dedicated temporal action recognition model would be more accurate:

- **Model options:** VideoMAE, TimeSformer, or SlowFast Networks
- **Capabilities:** Classify continuous activities from video clips (10–30 frames) rather than single frames
- **New activities detected:** exercising, cooking, video-calling, reading, stretching, dancing

This would eliminate false positives from object presence alone (e.g., a laptop on a desk doesn't mean the user is working — they might be watching a movie).

### 13.4 Predictive Behavior and Anticipatory Actions

By learning robust daily patterns, Aura could become genuinely anticipatory:

- Automatically dim LEDs and reduce notification frequency when the user's historical data predicts a focus session
- Prepare a "welcome home" greeting before the user arrives, based on typical arrival patterns
- Proactively remind the user of inferred upcoming needs ("You usually have lunch around now — the kitchen is just around the corner")

**Implementation:** Sequence modeling (LSTM or Transformer) over historical session data to predict next activity state.

### 13.5 Improved Personalization with Reinforcement Learning from Human Feedback (RLHF)

Current preference learning uses simple weight updates. A more sophisticated approach:

- Use **Direct Preference Optimization (DPO)** to fine-tune the agent's behavior model on pairs of (chosen, rejected) agent responses
- Collect implicit feedback labels from behavioral cues (did the user engage or dismiss?)
- Periodically retrain a lightweight adapter on accumulated preference data
- This enables Aura's personality to genuinely evolve toward each user's specific preferences over months of use

### 13.6 Multi-Camera and Depth Sensing

A single camera provides 2D perception. Adding:

- **Stereo camera or depth sensor (Intel RealSense, OAK-D):** Enables accurate 3D body tracking, true proximity measurement, fall detection
- **Wide-angle secondary camera:** Full room coverage, not just desk-facing view
- **Thermal sensor (optional):** Passive presence detection even in darkness, health monitoring

### 13.7 Smart Home Integration

Aura's contextual awareness makes it a natural smart home controller:

- When user sits down at desk → Automatically adjust smart lights to task lighting profile
- When user appears to be sleeping on couch → Slowly dim all lights, lower thermostat
- When a group is detected → Activate social lighting preset, disable work-mode notifications
- Integration via Home Assistant local API or Matter protocol

### 13.8 Privacy and On-Device Security Improvements

Long-term roadmap for privacy hardening:

- **Federated learning:** Share learned preference updates (not raw data) across a fleet of devices to improve base models without centralized data collection
- **Differential privacy:** Add calibrated noise to any telemetry before transmission
- **Encrypted face database:** Store face embeddings encrypted at rest with user-controlled key
- **Audit logging:** Full transparency log of every perception event and action decision, accessible to the user
- **Hardware privacy switch:** Physical camera shutter + microphone disconnect button

### 13.9 Emotional Intelligence Expansion

Current emotion detection covers 7 basic emotions. Future enhancements:

- **Valence-arousal-dominance (VAD) model:** Continuous 3-dimensional emotional space rather than discrete categories
- **Micro-expression detection:** Catch brief, suppressed emotional signals (disgust flashes, fear microexpressions)
- **Body language fusion:** Combine facial emotion with posture, gesture, and gait for a holistic emotional estimate
- **Contextual emotion calibration:** Recognize that the same expression means different things in different contexts (furrowed brow during coding = concentration, not anger)

---

*Document Version 1.0 — AI Companion Device (Aura) Technical Architecture*  
*Generated for engineering team implementation reference.*  
*All code samples are illustrative. Production implementation requires testing, error handling, and hardware-specific tuning.*
