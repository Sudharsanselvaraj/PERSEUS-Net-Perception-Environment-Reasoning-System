"""
scripts/enroll.py
─────────────────────────────────────────────────────────────────
Standalone face enrollment script.
Run this before starting the main pipeline to register users.

Usage:
    python scripts/enroll.py
    python scripts/enroll.py --user-id user_002 --name "Jordan" --frames 10
"""

import sys
import argparse
sys.path.insert(0, ".")  # Allow imports from project root

import cv2
from config.config import load_config
from perception.camera.capture import CameraInputLayer
from perception.camera.processor import FrameProcessor
from perception.recognition.face_recognizer import FaceRecognitionSystem
from personalization.user_profile import ProfileStore


def parse_args():
    p = argparse.ArgumentParser(description="Aura Face Enrollment")
    p.add_argument("--user-id", help="User ID (e.g. user_001)")
    p.add_argument("--name", help="Display name (e.g. Alex)")
    p.add_argument("--frames", type=int, default=10,
                   help="Number of frames to capture (default: 10)")
    p.add_argument("--config", default="config/settings.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    user_id   = args.user_id or input("Enter user ID (e.g. user_001): ").strip()
    user_name = args.name    or input("Enter display name: ").strip()
    n_frames  = args.frames

    print(f"\n[ Enrolling: {user_name} ({user_id}) ]")
    print(f"Will capture {n_frames} frames. Press SPACE to capture, Q to quit early.\n")

    camera = CameraInputLayer(cfg.camera)
    processor = FrameProcessor(cfg.frame)
    face_system = FaceRecognitionSystem(cfg.recognition)
    profile_store = ProfileStore(cfg.memory.profile_db_path)

    if not camera.initialize():
        print("ERROR: Could not open camera")
        sys.exit(1)
    face_system.initialize()
    camera.start()

    enrollment_frames = []

    try:
        while len(enrollment_frames) < n_frames:
            raw = camera.get_frame(timeout=1.0)
            if raw is None:
                continue

            processed = processor.process(raw)
            display = processed.bgr.copy()

            # Overlay instructions
            remaining = n_frames - len(enrollment_frames)
            cv2.putText(display, f"Frames remaining: {remaining}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "SPACE=capture  Q=quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Enrollment", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                enrollment_frames.append(processed.bgr)
                print(f"  ✓ Frame {len(enrollment_frames)}/{n_frames} captured")
            elif key == ord("q"):
                print("  Enrollment stopped early by user")
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    if len(enrollment_frames) < 3:
        print(f"\n✗ Not enough frames (need at least 3, got {len(enrollment_frames)})")
        sys.exit(1)

    print(f"\nProcessing {len(enrollment_frames)} frames...")
    success = face_system.enroll_user(user_id, user_name, enrollment_frames)

    if success:
        profile_store.get_or_create(user_id, user_name)
        print(f"✓ Successfully enrolled {user_name} ({user_id})")
        print(f"  Profile saved to: {cfg.memory.profile_db_path}")
    else:
        print("✗ Enrollment failed — no face detected in captured frames")
        sys.exit(1)


if __name__ == "__main__":
    main()
