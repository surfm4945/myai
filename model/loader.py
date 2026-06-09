"""
model/loader.py - Local AI Model Loader
Supports DialoGPT (when torch is available) with graceful fallback
to template-based generation using TF-IDF retrieval.
"""

import os
import logging
import re
import random
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Template responses for when the model is unavailable
TEMPLATES = {
    "question": [
        "That's an interesting question. Based on my knowledge, {topic} involves several important aspects worth exploring.",
        "Great question about {topic}! Here's what I know: the key concepts include understanding the fundamentals and applying them practically.",
        "Regarding {topic}: this is a broad area with many nuances. The most important thing to understand is the core principles involved.",
    ],
    "greeting": [
        "Hello! I'm ready to help you today.",
        "Hi there! What would you like to discuss?",
        "Hey! Great to chat with you. What's on your mind?",
    ],
    "default": [
        "I understand you're asking about {input}. This is an interesting topic. Could you tell me more about what specifically you'd like to know?",
        "That's a thoughtful point about {input}. I'd be happy to explore this further with you.",
        "Interesting! Regarding {input} — there are multiple perspectives to consider here.",
        "I'm processing your message about {input}. This touches on some fascinating ideas.",
    ],
    "thanks": [
        "You're welcome! Happy to help anytime.",
        "Glad I could assist! Feel free to ask anything else.",
        "My pleasure! Is there anything else you'd like to explore?",
    ],
    "farewell": [
        "Goodbye! It was a pleasure chatting with you.",
        "See you next time! Take care.",
        "Bye! Come back whenever you want to chat.",
    ],
}


def _classify_input(text: str) -> str:
    """Classify user input type for template selection."""
    text_lower = text.lower().strip()
    
    greetings = {"hello", "hi", "hey", "howdy", "greetings", "sup", "what's up"}
    farewells = {"bye", "goodbye", "see you", "later", "farewell", "take care"}
    thanks = {"thank you", "thanks", "thank", "thx", "appreciate"}
    
    if any(word in text_lower for word in greetings):
        return "greeting"
    if any(word in text_lower for word in farewells):
        return "farewell"
    if any(word in text_lower for word in thanks):
        return "thanks"
    if text_lower.endswith("?") or text_lower.startswith(("what", "how", "why", "when", "where", "who", "which", "can", "could", "would", "is", "are", "do", "does")):
        return "question"
    return "default"


def _extract_topic(text: str) -> str:
    """Extract main topic from user input."""
    stop_words = {"what", "is", "a", "an", "the", "how", "why", "when", "where", "who", "are", "do", "does", "can", "could", "would", "tell", "me", "about", "explain"}
    words = text.lower().split()
    topic_words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(topic_words[:5]) if topic_words else text[:30]


class ModelLoader:
    """Local AI model loader with graceful fallback."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "microsoft/DialoGPT-small"
        self._loaded = False
        self._error = None
        self._use_torch = False

    def load(self) -> bool:
        """Load the DialoGPT model. Falls back gracefully if torch unavailable."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            )
            self.model.eval()
            self._loaded = True
            self._use_torch = True
            logger.info("DialoGPT model loaded successfully")
            return True

        except ImportError:
            logger.warning("torch not available — using template-based generation")
            self._loaded = True  # Mark as "loaded" so the app works
            return True
        except Exception as e:
            self._error = str(e)
            logger.warning(f"Model load failed ({e}) — using template-based generation")
            self._loaded = True  # Still mark loaded for fallback mode
            return True

    def is_loaded(self) -> bool:
        return self._loaded

    def is_neural(self) -> bool:
        """Returns True if using actual neural model."""
        return self._use_torch and self.model is not None

    def generate(self, user_input: str, history: Optional[List[Dict]] = None) -> str:
        """Generate a response. Uses DialoGPT if available, else templates."""
        if not self._loaded:
            return "Model not ready yet. Please wait a moment and try again."

        if self._use_torch and self.model is not None:
            return self._generate_neural(user_input, history)
        else:
            return self._generate_template(user_input, history)

    def is_quality(self, text: str) -> bool:
        """Return True if a generated response is clean enough to show."""
        if not text or len(text.strip()) < 15:
            return False

        t = text.lower()

        # Reddit / internet slang markers DialoGPT loves to emit
        reddit_markers = [
            "lol", "lmao", "xd", "rofl", "haha", "hehe",
            "op ", "cakeday", "/r/", "upvote", "downvote",
            "edit:", "formatting", "jk", ":p", ":)", ";)",
            "omg", "smh", "tbh", "ngl", "imo", "idk",
        ]
        hits = sum(1 for m in reddit_markers if m in t)
        if hits >= 2:
            return False

        # Excess punctuation (excited/spammy)
        if text.count("!") > 3 or text.count("?") > 3:
            return False

        # Contains URLs
        if "http" in t or "www." in t:
            return False

        # Very repetitive (e.g., "haha haha haha")
        words = t.split()
        if len(words) > 3 and len(set(words)) / len(words) < 0.5:
            return False

        return True

    def safe_template(self, user_input: str) -> str:
        """Return a clean template response (never neural)."""
        return self._generate_template(user_input)

    def generate_with_context(self, user_input: str, brain_context: str,
                               history: Optional[List[Dict]] = None) -> str:
        """Return brain_context directly — it's curated and always clean."""
        return brain_context

    # ── Neural generation ──────────────────────────────────────────────────────

    def _generate_neural(self, user_input: str, history: Optional[List[Dict]] = None) -> str:
        """Generate with DialoGPT."""
        try:
            import torch

            # Build context string from recent history
            context = ""
            if history:
                for msg in history[-4:]:  # Last 4 messages for context window
                    role = "User" if msg["role"] == "user" else "Assistant"
                    content = msg["content"][:200]  # Truncate long messages
                    context += f"{role}: {content}\n"
            context += f"User: {user_input}\nAssistant:"

            input_ids = self.tokenizer.encode(
                context + self.tokenizer.eos_token,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=120,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.75,
                    top_p=0.92,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=4,
                )

            new_tokens = output_ids[:, input_ids.shape[-1]:]
            response = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
            return self._clean(response) or self._generate_template(user_input, history)

        except Exception as e:
            logger.error(f"Neural generation error: {e}")
            return self._generate_template(user_input, history)

    # ── Template generation ────────────────────────────────────────────────────

    def _generate_template(self, user_input: str, history: Optional[List[Dict]] = None) -> str:
        """Generate a template-based response."""
        kind = _classify_input(user_input)
        topic = _extract_topic(user_input)
        templates = TEMPLATES.get(kind, TEMPLATES["default"])
        template = random.choice(templates)
        try:
            return template.format(topic=topic, input=user_input[:60])
        except KeyError:
            return template

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        """Clean generated text."""
        if not text:
            return ""

        # Remove repetitive lines
        lines = text.split("\n")
        seen: set = set()
        clean: list = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in seen:
                clean.append(line)
                seen.add(stripped)

        result = " ".join(clean).strip()

        # Trim to last complete sentence
        if result and result[-1] not in ".!?":
            last = max(result.rfind("."), result.rfind("!"), result.rfind("?"))
            if last > len(result) // 3:
                result = result[: last + 1]

        return result
