"""
utils/memory.py - Chat Session Memory
Manages persistent chat sessions stored as JSON files.
"""

import json
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ChatMemory:
    """Manages chat sessions on disk as JSON files."""

    def __init__(self, sessions_dir: str):
        self.sessions_dir = sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())[:8]
        session = {
            "id": session_id,
            "title": "New Chat",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
        }
        self._save(session)
        return session_id

    def delete_session(self, session_id: str):
        """Delete a session file."""
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)

    def clear_all(self):
        """Delete all session files."""
        for fname in os.listdir(self.sessions_dir):
            if fname.endswith(".json"):
                os.remove(os.path.join(self.sessions_dir, fname))

    # ── Message management ─────────────────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str):
        """Append a message to a session."""
        session = self._load(session_id)
        if session is None:
            logger.warning(f"Session {session_id} not found")
            return

        session["messages"].append(
            {
                "id": str(uuid.uuid4())[:8],
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )
        session["updated_at"] = datetime.now().isoformat()
        self._save(session)

    def get_messages(self, session_id: str) -> List[Dict]:
        """Return all messages for a session."""
        session = self._load(session_id)
        return session.get("messages", []) if session else []

    def update_session_title(self, session_id: str, title: str):
        """Update a session's display title."""
        session = self._load(session_id)
        if session:
            session["title"] = title[:60]
            self._save(session)

    # ── Listing ────────────────────────────────────────────────────────────────

    def list_sessions(self) -> List[Dict]:
        """
        Return all sessions sorted newest first.
        Each entry: id, title, created_at, updated_at, message_count.
        """
        sessions = []
        for fname in os.listdir(self.sessions_dir):
            if not fname.endswith(".json"):
                continue
            session = self._load(fname[:-5])
            if session:
                sessions.append(
                    {
                        "id": session["id"],
                        "title": session.get("title", "New Chat"),
                        "created_at": session.get("created_at", ""),
                        "updated_at": session.get("updated_at", ""),
                        "message_count": len(session.get("messages", [])),
                    }
                )

        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions

    def get_session(self, session_id: str) -> Optional[Dict]:
        return self._load(session_id)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _path(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, f"{session_id}.json")

    def _load(self, session_id: str) -> Optional[Dict]:
        path = self._path(session_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None

    def _save(self, session: Dict):
        path = self._path(session["id"])
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving session: {e}")
