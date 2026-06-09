"""
model/trainer.py - Brain Trainer
Manages the brain.kesar persistent memory file.
Handles adding entries, updating frequencies, and stats.
"""

import json
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BrainTrainer:
    """Manages the AI's persistent knowledge base (brain.kesar)."""

    def __init__(self, brain_path: str):
        self.brain_path = brain_path
        self._data: Dict = {}
        self._load()

    # ── I/O ────────────────────────────────────────────────────────────────────

    def _load(self):
        """Load brain.kesar, creating it if it doesn't exist."""
        try:
            if os.path.exists(self.brain_path):
                with open(self.brain_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(f"Brain loaded: {len(self._data.get('entries', []))} entries")
            else:
                logger.info("No brain file found — creating empty brain")
                self._data = self._empty_brain()
                self._save()
        except Exception as e:
            logger.error(f"Error loading brain: {e}")
            self._data = self._empty_brain()

    def _save(self):
        """Persist brain.kesar to disk."""
        try:
            self._data["updated_at"] = datetime.now().isoformat()
            self._data["stats"]["total_entries"] = len(self._data.get("entries", []))
            os.makedirs(os.path.dirname(self.brain_path), exist_ok=True)
            with open(self.brain_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving brain: {e}")

    def _empty_brain(self) -> Dict:
        return {
            "version": "1.0",
            "description": "AI Brain - Persistent Memory Store",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "stats": {
                "total_conversations": 0,
                "total_messages": 0,
                "total_entries": 0,
            },
            "entries": [],
        }

    # ── Entry management ───────────────────────────────────────────────────────

    def add_entry(self, question: str, answer: str, tags: Optional[List[str]] = None) -> str:
        """
        Add or update a Q&A entry.
        Returns the entry ID (existing or new).
        """
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return ""

        # Update existing entry with same question
        for entry in self._data.setdefault("entries", []):
            if entry["question"].lower() == question.lower():
                entry["answer"] = answer
                entry["frequency"] = entry.get("frequency", 0) + 1
                entry["last_used"] = datetime.now().isoformat()
                self._save()
                return entry["id"]

        # Create new entry (limit total to 2000 to avoid bloat)
        if len(self._data["entries"]) >= 2000:
            # Remove least-used entry
            self._data["entries"].sort(key=lambda e: e.get("frequency", 0))
            self._data["entries"].pop(0)

        entry_id = str(uuid.uuid4())[:8]
        entry = {
            "id": entry_id,
            "question": question,
            "answer": answer,
            "tags": tags or [],
            "frequency": 1,
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
        }
        self._data["entries"].append(entry)
        self._save()
        return entry_id

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        before = len(self._data.get("entries", []))
        self._data["entries"] = [e for e in self._data.get("entries", []) if e["id"] != entry_id]
        if len(self._data["entries"]) < before:
            self._save()
            return True
        return False

    def increment_frequency(self, entry_id: str):
        """Increment usage counter for an entry."""
        for entry in self._data.get("entries", []):
            if entry["id"] == entry_id:
                entry["frequency"] = entry.get("frequency", 0) + 1
                entry["last_used"] = datetime.now().isoformat()
                self._save()
                return

    def get_entries(self) -> List[Dict]:
        return self._data.get("entries", [])

    def get_brain_data(self) -> Dict:
        return self._data

    def increment_stats(self, messages: int = 2):
        """Bump conversation and message counters."""
        stats = self._data.setdefault("stats", {})
        stats["total_conversations"] = stats.get("total_conversations", 0) + 1
        stats["total_messages"] = stats.get("total_messages", 0) + messages
        self._save()

    # ── Search ─────────────────────────────────────────────────────────────────

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple keyword-based fallback search."""
        query_words = set(query.lower().split())
        scored = []
        for entry in self._data.get("entries", []):
            entry_words = set(entry["question"].lower().split())
            overlap = len(query_words & entry_words)
            if overlap:
                score = overlap / max(len(query_words), len(entry_words))
                scored.append({"entry": entry, "similarity": score})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]
