"""
main.py
─────────────────────────────────────────────────────────────────
Aura AI Companion Device — Main Pipeline Entry Point

Wires together all components and runs the main perception loop.

Usage:
    python main.py                        # Run with config/settings.yaml
    python main.py --config my_config.yaml
    python main.py --enroll               # Run enrollment wizard
    python main.py --no-vlm               # Disable VLM for faster startup
    python main.py --show-video           # Show live annotated video window
"""

from __future__ import annotations

import signal
import sys
import time
import argparse

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────
from config.config import load_config, get_config

# ── Logging ───────────────────────────────────────────────────
from utils.logger import setup_logging, get_logger
from utils.timing import profiler

# ── Perception ────────────────────────────────────────────────
from perception.camera.capture import CameraInputLayer
from perception.camera.processor import FrameProcessor
from perception.detection.human_detector import HumanDetector
from perception.recognition.face_recognizer import FaceRecognitionSystem
from perception.emotion.emotion_detector import EmotionDetector
from perception.gesture.gesture_recognizer import GestureRecognizer
from perception.objects.object_detector import ContextObjectDetector
from perception.scene.vlm_analyzer import VLMSceneAnalyzer
from perception.orchestrator import PerceptionOrchestrator

# ── Memory ────────────────────────────────────────────────────
from memory.temporal_buffer import TemporalMemoryBuffer, PerceptionSnapshot
from memory.session_memory import SessionMemory

# ── Context ───────────────────────────────────────────────────
from context.context_engine import ContextEngine

# ── Personalization ───────────────────────────────────────────
from personalization.user_profile import ProfileStore
from personalization.personalization_engine import PersonalizationEngine

# ── Agent ─────────────────────────────────────────────────────
from agent.aura_agent import AuraAgent

# ── Behavior ─────────────────────────────────────────────────
from behavior.tts.tts_engine import TTSEngine
from behavior.leds.led_controller import LEDController
from behavior.executor import BehaviorExecutor

# ── Hardware ─────────────────────────────────────────────────
from hardware.microcontroller import MicrocontrollerBridge

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Aura AI Companion Device")
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Path to settings YAML")
    parser.add_argument("--enroll", action="store_true",
                        help="Run face enrollment wizard")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Disable VLM scene analysis")
    parser.add_argument("--show-video", action="store_true",
                        help="Show live annotated video window")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run perception only — skip TTS and LEDs")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Component Factory
# ─────────────────────────────────────────────────────────────

def build_components(args):
    """Instantiate and initialize all system components."""
    cfg = load_config(args.config)
    setup_logging(cfg.app.log_level, cfg.app.log_dir, cfg.app.debug)

    logger.info(f"Starting {cfg.app.name} v{cfg.app.version}")

    if args.no_vlm:
        cfg.scene.enabled = False

    # Perception
    camera = CameraInputLayer(cfg.camera)
    processor = FrameProcessor(cfg.frame)
    detector = HumanDetector(cfg.detection)
    face_recognizer = FaceRecognitionSystem(cfg.recognition)
    emotion_detector = EmotionDetector(cfg.emotion)
    gesture_recognizer = GestureRecognizer(cfg.gesture)
    object_detector = ContextObjectDetector(cfg.objects)
    scene_analyzer = VLMSceneAnalyzer(cfg.scene)

    # Memory
    short_term_buffer = TemporalMemoryBuffer(cfg.memory.short_term_window_seconds)
    session = SessionMemory()

    # Context + Personalization
    context_engine = ContextEngine()
    profile_store = ProfileStore(cfg.memory.profile_db_path)
    personalization = PersonalizationEngine(profile_store)

    # Agent
    agent = AuraAgent(cfg.agent, cfg.cooldowns)

    # Behavior
    tts = TTSEngine(cfg.behavior) if not args.dry_run else None
    leds = LEDController(cfg.hardware)
    executor = BehaviorExecutor(tts or _NullTTS(), leds) if not args.dry_run else None

    # Hardware
    mcu = MicrocontrollerBridge(cfg.hardware)

    return {
        "cfg": cfg,
        "camera": camera,
        "processor": processor,
        "detector": detector,
        "face_recognizer": face_recognizer,
        "emotion_detector": emotion_detector,
        "gesture_recognizer": gesture_recognizer,
        "object_detector": object_detector,
        "scene_analyzer": scene_analyzer,
        "short_term_buffer": short_term_buffer,
        "session": session,
        "context_engine": context_engine,
        "profile_store": profile_store,
        "personalization": personalization,
        "agent": agent,
        "tts": tts,
        "leds": leds,
        "executor": executor,
        "mcu": mcu,
        "args": args,
    }


# ─────────────────────────────────────────────────────────────
# Enrollment Wizard
# ─────────────────────────────────────────────────────────────

def run_enrollment_wizard(components: dict) -> None:
    cfg = components["cfg"]
    camera: CameraInputLayer = components["camera"]
    face_recognizer: FaceRecognitionSystem = components["face_recognizer"]
    profile_store: ProfileStore = components["profile_store"]

    camera.initialize()
    camera.start()
    face_recognizer.initialize()

    print("\n─── Aura Face Enrollment Wizard ───────────────")
    user_id   = input("Enter user ID (e.g. 'user_001'): ").strip()
    user_name = input("Enter display name (e.g. 'Alex'): ").strip()

    print(f"\nCapturing enrollment frames for '{user_name}'.")
    print("Look at the camera. Press SPACE to capture, Q to finish.\n")

    enrollment_frames = []
    processor = FrameProcessor(cfg.frame)

    while True:
        raw = camera.get_frame(timeout=1.0)
        if raw is None:
            continue

        processed = processor.process(raw)
        cv2.imshow("Enrollment — press SPACE to capture, Q to quit", processed.bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            enrollment_frames.append(processed.bgr)
            print(f"  Captured frame {len(enrollment_frames)}")
        elif key == ord('q') or len(enrollment_frames) >= 15:
            break

    cv2.destroyAllWindows()
    camera.stop()

    if enrollment_frames:
        success = face_recognizer.enroll_user(user_id, user_name, enrollment_frames)
        if success:
            profile_store.get_or_create(user_id, user_name)
            print(f"\n✓ Enrolled {user_name} ({user_id}) successfully.")
        else:
            print("\n✗ Enrollment failed — no face detected in frames.")
    else:
        print("\nNo frames captured — enrollment cancelled.")


# ─────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────

class AuraSystem:
    """Top-level system controller."""

    def __init__(self, components: dict):
        self.c = components
        self._running = False
        self._last_context = None
        self._frame_count = 0

    def initialize(self) -> None:
        c = self.c
        cfg = c["cfg"]

        # Initialize hardware
        c["mcu"].connect()
        c["leds"].initialize()
        if c["tts"]:
            c["tts"].initialize()

        # Initialize camera
        if not c["camera"].initialize():
            logger.error("Camera initialization failed — cannot start")
            sys.exit(1)
        c["camera"].start()

        # Initialize perception models
        orchestrator = PerceptionOrchestrator(
            config=cfg.pipeline,
            detector=c["detector"],
            face_recognizer=c["face_recognizer"],
            emotion_detector=c["emotion_detector"],
            gesture_recognizer=c["gesture_recognizer"],
            object_detector=c["object_detector"],
            scene_analyzer=c["scene_analyzer"],
            on_tier3_complete=self._on_tier3_complete,
        )
        orchestrator.initialize()
        self.orchestrator = orchestrator

        logger.info("Aura system fully initialized — starting main loop")

    def run(self) -> None:
        self._running = True
        args = self.c["args"]
        processor: FrameProcessor = self.c["processor"]

        while self._running:
            raw_frame = self.c["camera"].get_frame(timeout=0.1)
            if raw_frame is None:
                continue

            processed = processor.process(raw_frame)
            if not processed.metadata.is_usable:
                continue

            self._frame_count += 1

            # Push frame through orchestrator
            state = self.orchestrator.process_frame(processed.bgr)

            # Update short-term memory buffer every frame
            self._update_memory(state, processed.metadata.timestamp)

            # Optionally show annotated video
            if args.show_video:
                self._render_debug_frame(processed.bgr, state)

        logger.info("Main loop exited")

    def _update_memory(self, state, timestamp: float) -> None:
        buffer: TemporalMemoryBuffer = self.c["short_term_buffer"]
        face_result = (state.face_results[0] if state.face_results else None)
        gesture = state.gesture

        snapshot = PerceptionSnapshot(
            timestamp=timestamp,
            user_id=face_result.user_id if face_result else None,
            emotion=state.emotion.dominant_emotion if state.emotion else None,
            gesture=gesture.gesture_name if gesture else "none",
            detected_objects=state.objects.detected_objects if state.objects else [],
            activity=state.objects.inferred_activity if state.objects else "idle",
            body_orientation=gesture.body_orientation if gesture else "unknown",
            human_present=state.human_present,
            face_confidence=face_result.confidence if face_result else 0.0,
        )
        buffer.add(snapshot)

    def _on_tier3_complete(self) -> None:
        """Called from background thread after VLM analysis — build context + agent decision."""
        try:
            c = self.c
            state = self.orchestrator.get_state()
            buffer_summary = c["short_term_buffer"].summarize()

            context = c["context_engine"].build_context(
                face_results=state.face_results,
                emotion=state.emotion,
                gesture=state.gesture,
                objects=state.objects,
                scene=state.scene,
                buffer_summary=buffer_summary,
                session=c["session"],
            )
            self._last_context = context
            logger.debug(f"Context built: event={context.event}, "
                         f"emotion={context.emotion}, activity={context.activity}")

            # Personalization
            profile = c["personalization"].get_profile_for_context(context)
            behavior_instructions = c["personalization"].get_behavior_instructions(
                profile, context
            )

            # Agent decision
            action = c["agent"].decide(context, behavior_instructions)
            logger.info(f"Agent action: {action.action_type} — {action.reasoning}")

            # Execute behavior
            if c["executor"] and action.action_type != "silence":
                c["executor"].execute(action)
                if context.user_id:
                    c["session"].log_interaction(
                        action.action_type, action.message or "", action.tone
                    )
        except Exception as e:
            logger.error(f"Error in _on_tier3_complete: {e}", exc_info=True)

    def _render_debug_frame(self, frame: np.ndarray, state) -> None:
        """Draw bounding boxes and labels on frame for debugging."""
        debug = frame.copy()
        for det in state.human_detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {det.confidence:.0%}"
            cv2.putText(debug, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if state.emotion:
            cv2.putText(debug, f"Emotion: {state.emotion.dominant_emotion}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        if state.gesture:
            cv2.putText(debug, f"Gesture: {state.gesture.gesture_name}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        cv2.imshow("Aura — Debug View (press Q to quit)", debug)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop()

    def stop(self) -> None:
        self._running = False
        self.c["camera"].stop()
        self.orchestrator.shutdown()
        if self.c["tts"]:
            self.c["tts"].stop()
        self.c["leds"].off()
        self.c["mcu"].disconnect()
        cv2.destroyAllWindows()
        logger.info(profiler.report())
        logger.info("Aura shutdown complete")


class _NullTTS:
    """No-op TTS for dry-run mode."""
    def speak(self, text, tone="neutral", blocking=False): pass
    def is_speaking(self): return False
    def stop(self): pass


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    components = build_components(args)

    if args.enroll:
        run_enrollment_wizard(components)
        return

    system = AuraSystem(components)

    def _signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    system.initialize()
    system.run()


if __name__ == "__main__":
    main()
