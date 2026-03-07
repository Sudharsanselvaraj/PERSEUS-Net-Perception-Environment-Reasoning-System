"""
perception/camera/capture.py
─────────────────────────────────────────────────────────────────
Thread-safe camera capture layer.
Runs a dedicated background thread for frame acquisition so the
main pipeline thread always picks up the freshest frame without
being blocked by camera I/O latency.
"""

from __future__ import annotations

import time
import threading
from queue import Queue, Empty
from typing import Optional

import cv2
import numpy as np

from config.config import CameraConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class CameraInputLayer:
    """
    Continuous video capture with a bounded queue.
    The capture thread always drops the oldest frame when the
    queue is full, guaranteeing freshness for downstream consumers.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: Queue[np.ndarray] = Queue(maxsize=2)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._last_fps_ts = time.time()
        self._fps_estimate = 0.0

    # ── Lifecycle ─────────────────────────────────────────────

    def initialize(self) -> bool:
        """Open the camera device and configure capture properties."""
        backend_map = {
            "v4l2": cv2.CAP_V4L2,
            "dshow": cv2.CAP_DSHOW,
            "avfoundation": cv2.CAP_AVFOUNDATION,
            "any": cv2.CAP_ANY,
        }
        backend = backend_map.get(self.config.backend.lower(), cv2.CAP_ANY)

        self.cap = cv2.VideoCapture(self.config.device_id, backend)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera device {self.config.device_id}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        codec_chars = self.config.codec
        if len(codec_chars) == 4:
            fourcc = cv2.VideoWriter_fourcc(*codec_chars)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {actual_w}×{actual_h} @ {actual_fps:.1f} FPS "
                    f"(device {self.config.device_id})")
        return True

    def start(self) -> None:
        """Start background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CaptureThread",
            daemon=True
        )
        self._thread.start()
        logger.info("Camera capture thread started")

    def stop(self) -> None:
        """Stop capture thread and release camera."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("Camera capture thread stopped")

    # ── Frame Access ──────────────────────────────────────────

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get the latest frame from the queue.
        Returns None if no frame is available within `timeout` seconds.
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def fps_estimate(self) -> float:
        return self._fps_estimate

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal ──────────────────────────────────────────────

    def _capture_loop(self) -> None:
        consecutive_failures = 0
        while self._running:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    logger.error("Camera capture failed 30 consecutive times — stopping")
                    self._running = False
                time.sleep(0.01)
                continue

            consecutive_failures = 0
            self._frame_count += 1

            # FPS estimation every 60 frames
            if self._frame_count % 60 == 0:
                now = time.time()
                elapsed = now - self._last_fps_ts
                self._fps_estimate = 60.0 / elapsed if elapsed > 0 else 0.0
                self._last_fps_ts = now

            # Drop oldest if queue is full — always serve freshest
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass

            self._frame_queue.put(frame)
