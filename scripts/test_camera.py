"""
scripts/test_camera.py
─────────────────────────────────────────────────────────────────
Camera diagnostic utility.
Verifies camera access, resolution, and frame quality.

Usage:
    python scripts/test_camera.py
    python scripts/test_camera.py --device 1
"""

import sys
import time
import argparse
sys.path.insert(0, ".")

import cv2
from config.config import load_config
from perception.camera.capture import CameraInputLayer
from perception.camera.processor import FrameProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--config", default="config/settings.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.camera.device_id = args.device

    camera = CameraInputLayer(cfg.camera)
    processor = FrameProcessor(cfg.frame)

    print(f"Testing camera device {args.device}...")
    if not camera.initialize():
        print("ERROR: Camera not available")
        sys.exit(1)

    camera.start()
    time.sleep(0.5)

    start = time.time()
    frame_count = 0

    print("Press Q to quit diagnostic view\n")
    while True:
        raw = camera.get_frame(timeout=0.5)
        if raw is None:
            continue

        processed = processor.process(raw)
        frame_count += 1
        elapsed = time.time() - start
        fps = frame_count / elapsed

        # Annotate
        display = processed.bgr.copy()
        m = processed.metadata
        color = (0, 255, 0) if m.is_usable else (0, 0, 255)
        status = "USABLE" if m.is_usable else "POOR QUALITY"

        cv2.putText(display, f"{status} | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Brightness: {m.brightness:.0f} | Blur: {m.blur_score:.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, f"Resolution: {m.processed_shape[1]}x{m.processed_shape[0]}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Camera Diagnostic", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.stop()
    cv2.destroyAllWindows()
    print(f"\nAverage FPS: {frame_count / (time.time() - start):.1f}")


if __name__ == "__main__":
    main()
