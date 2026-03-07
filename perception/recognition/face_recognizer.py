"""
perception/recognition/face_recognizer.py
─────────────────────────────────────────────────────────────────
InsightFace / ArcFace face recognition system.
Tier-2 model: runs every ~3 seconds.
Maintains an on-disk enrollment database of face embeddings.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config.config import RecognitionConfig
from utils.logger import get_logger
from utils.timing import timed

logger = get_logger(__name__)


@dataclass
class FaceRecognitionResult:
    user_id: Optional[str]
    display_name: Optional[str]
    confidence: float          # Cosine similarity 0–1
    face_bbox: tuple           # (x1, y1, x2, y2)
    face_embedding: np.ndarray
    is_new_face: bool          # True if below any-match threshold


@dataclass
class EnrolledUser:
    user_id: str
    display_name: str
    embedding: np.ndarray
    enrolled_at: float = field(default_factory=time.time)
    enrollment_frame_count: int = 1


class FaceRecognitionSystem:
    """
    Wraps InsightFace for detection + embedding extraction.
    Persists user embeddings in a local pickle database.
    Uses cosine similarity for identity verification.
    """

    NEW_FACE_UPPER_THRESHOLD = 0.20   # Below this → truly new face

    def __init__(self, config: RecognitionConfig):
        self.config = config
        self._app = None
        self._initialized = False
        self._db: Dict[str, EnrolledUser] = {}
        self._load_database()

    # ── Lifecycle ─────────────────────────────────────────────

    def initialize(self) -> None:
        try:
            import insightface
            self._app = insightface.app.FaceAnalysis(
                name=self.config.model_name,
                allowed_modules=["detection", "recognition"],
            )
            self._app.prepare(
                ctx_id=self.config.ctx_id,
                det_size=tuple(self.config.det_size),
            )
            logger.info(f"FaceRecognitionSystem initialized: {self.config.model_name}")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize FaceRecognitionSystem: {e}")
            raise

    # ── Enrollment ────────────────────────────────────────────

    def enroll_user(self, user_id: str, display_name: str,
                    frames: List[np.ndarray]) -> bool:
        """
        Register a new user from a list of enrollment frames.
        Computes the mean embedding and L2-normalizes it.
        Minimum 3 frames recommended for reliable enrollment.
        """
        if not self._initialized:
            raise RuntimeError("System not initialized.")

        embeddings = []
        for frame in frames:
            faces = self._app.get(frame)
            if faces:
                emb = faces[0].embedding
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)

        if not embeddings:
            logger.warning(f"No faces detected during enrollment for {user_id}")
            return False

        avg_emb = np.mean(embeddings, axis=0)
        avg_emb /= np.linalg.norm(avg_emb)

        self._db[user_id] = EnrolledUser(
            user_id=user_id,
            display_name=display_name,
            embedding=avg_emb,
            enrollment_frame_count=len(embeddings),
        )
        self._save_database()
        logger.info(f"Enrolled user '{display_name}' ({user_id}) "
                    f"from {len(embeddings)} frames")
        return True

    def delete_user(self, user_id: str) -> bool:
        if user_id in self._db:
            del self._db[user_id]
            self._save_database()
            return True
        return False

    def list_users(self) -> List[str]:
        return [u.display_name for u in self._db.values()]

    # ── Recognition ───────────────────────────────────────────

    @timed("face_recognition")
    def recognize(self, frame: np.ndarray) -> List[FaceRecognitionResult]:
        """
        Detect and recognize all faces in `frame`.
        Returns one result per detected face.
        """
        if not self._initialized:
            raise RuntimeError("System not initialized.")

        faces = self._app.get(frame)
        results: List[FaceRecognitionResult] = []

        for face in faces:
            query_emb = face.embedding / np.linalg.norm(face.embedding)
            best_id: Optional[str] = None
            best_name: Optional[str] = None
            best_score = 0.0

            for enrolled in self._db.values():
                score = float(np.dot(query_emb, enrolled.embedding))
                if score > best_score:
                    best_score = score
                    best_id = enrolled.user_id
                    best_name = enrolled.display_name

            is_recognized = best_score >= self.config.recognition_threshold
            is_new = (not is_recognized and best_score < self.NEW_FACE_UPPER_THRESHOLD)
            bbox = tuple(face.bbox.astype(int).tolist())

            results.append(FaceRecognitionResult(
                user_id=best_id if is_recognized else None,
                display_name=best_name if is_recognized else None,
                confidence=best_score,
                face_bbox=bbox,
                face_embedding=query_emb,
                is_new_face=is_new,
            ))

        return results

    # ── Persistence ───────────────────────────────────────────

    def _load_database(self) -> None:
        path = Path(self.config.database_path)
        if path.exists():
            with open(path, "rb") as f:
                self._db = pickle.load(f)
            logger.info(f"Loaded {len(self._db)} enrolled users from {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("No existing face database found — starting fresh")

    def _save_database(self) -> None:
        path = Path(self.config.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._db, f)
