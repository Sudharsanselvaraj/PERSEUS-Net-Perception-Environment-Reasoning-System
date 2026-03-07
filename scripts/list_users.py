"""
scripts/list_users.py
─────────────────────────────────────────────────────────────────
List all enrolled users and their profile details.

Usage:
    python scripts/list_users.py
"""

import sys
import json
sys.path.insert(0, ".")

from config.config import load_config
from perception.recognition.face_recognizer import FaceRecognitionSystem
from personalization.user_profile import ProfileStore


def main():
    cfg = load_config()
    face_system = FaceRecognitionSystem(cfg.recognition)
    profile_store = ProfileStore(cfg.memory.profile_db_path)

    face_system.initialize()
    users = profile_store.list_users()

    if not users:
        print("No users enrolled yet. Run: python scripts/enroll.py")
        return

    print(f"\n─── Enrolled Users ({len(users)}) ──────────────────\n")
    for user_id in users:
        profile = profile_store.load(user_id)
        if profile:
            print(f"  {profile.display_name} ({profile.user_id})")
            print(f"    Style: {profile.preferred_interaction_style}")
            print(f"    Reminders: {profile.reminder_tolerance}")
            print(f"    Interactions: +{profile.positive_interactions} / -{profile.negative_interactions}")
            print()


if __name__ == "__main__":
    main()
