"""
utils/embeddings.py - Semantic Search Engine
Uses TF-IDF + cosine similarity (scikit-learn) as the primary engine,
with optional FAISS + sentence-transformers for richer embeddings.
Falls back gracefully when heavy dependencies aren't available.
"""

import json
import os
import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class EmbeddingSearch:
    """
    Semantic similarity search over brain.kesar entries.

    Priority order:
    1. FAISS + sentence-transformers (best quality)
    2. TF-IDF + cosine similarity (scikit-learn — always available)
    3. Keyword overlap (last resort)
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.entries: List[Dict] = []

        # Backend state
        self._faiss_index = None
        self._st_model = None
        self._tfidf_matrix = None
        self._tfidf_vectorizer = None

        self._has_faiss = False
        self._has_st = False
        self._has_sklearn = False

        self._init_backends()

    # ── Backend initialisation ─────────────────────────────────────────────────

    def _init_backends(self):
        """Detect and initialise available backends."""
        # sentence-transformers + FAISS
        try:
            import faiss  # noqa: F401
            self._has_faiss = True
            logger.info("FAISS available")
        except ImportError:
            pass

        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._has_st = True
            logger.info("sentence-transformers loaded")
        except Exception as e:
            logger.debug(f"sentence-transformers not available: {e}")

        # scikit-learn TF-IDF
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
            self._has_sklearn = True
            logger.info("scikit-learn TF-IDF available")
        except ImportError:
            pass

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_from_brain(self, brain_path: str):
        """Build search index from brain.kesar entries."""
        if not os.path.exists(brain_path):
            return
        try:
            with open(brain_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", [])
            if not entries:
                return
            self.entries = list(entries)
            self._rebuild_index()
        except Exception as e:
            logger.error(f"Error loading brain for embeddings: {e}")

    def add_entry(self, question: str, answer: str, entry_id: str):
        """Add a single entry and update the index incrementally."""
        # Avoid duplicates in entries list
        if any(e.get("id") == entry_id for e in self.entries):
            return
        self.entries.append({"id": entry_id, "question": question, "answer": answer})
        # Rebuild (incremental update is complex; full rebuild is fast for <2k entries)
        self._rebuild_index()

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Return top_k most similar entries.
        Each result: { "entry": {...}, "similarity": float }
        """
        if not self.entries:
            return []

        # Try best backend first
        if self._has_st and self._st_model is not None and self._has_faiss and self._faiss_index is not None:
            return self._search_faiss(query, top_k)
        if self._has_st and self._st_model is not None and len(self.entries) > 0:
            return self._search_st_numpy(query, top_k)
        if self._has_sklearn and self._tfidf_vectorizer is not None:
            return self._search_tfidf(query, top_k)
        return self._search_keyword(query, top_k)

    # ── Index builders ─────────────────────────────────────────────────────────

    def _rebuild_index(self):
        """Rebuild whichever index is best available."""
        if not self.entries:
            return
        questions = [e["question"] for e in self.entries]

        if self._has_st and self._st_model:
            try:
                embeddings = self._st_model.encode(questions, show_progress_bar=False)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                embeddings = embeddings / norms

                if self._has_faiss:
                    import faiss
                    dim = embeddings.shape[1]
                    self._faiss_index = faiss.IndexFlatIP(dim)
                    self._faiss_index.add(embeddings.astype("float32"))
                else:
                    self._st_embeddings = embeddings.astype("float32")
                return
            except Exception as e:
                logger.warning(f"sentence-transformers index build failed: {e}")

        if self._has_sklearn:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words="english",
                )
                self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(questions)
            except Exception as e:
                logger.warning(f"TF-IDF index build failed: {e}")

    # ── Search implementations ─────────────────────────────────────────────────

    def _search_faiss(self, query: str, top_k: int) -> List[Dict]:
        try:
            q_emb = self._st_model.encode([query], show_progress_bar=False)
            q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
            k = min(top_k, self._faiss_index.ntotal)
            distances, indices = self._faiss_index.search(q_emb.astype("float32"), k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.entries):
                    results.append({"entry": self.entries[idx], "similarity": float(dist)})
            return results
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return self._search_tfidf(query, top_k)

    def _search_st_numpy(self, query: str, top_k: int) -> List[Dict]:
        try:
            q_emb = self._st_model.encode([query], show_progress_bar=False)
            q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
            sims = np.dot(self._st_embeddings, q_emb.T).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [
                {"entry": self.entries[i], "similarity": float(sims[i])}
                for i in top_indices
                if i < len(self.entries)
            ]
        except Exception as e:
            logger.error(f"ST-numpy search error: {e}")
            return self._search_tfidf(query, top_k)

    def _search_tfidf(self, query: str, top_k: int) -> List[Dict]:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = self._tfidf_vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [
                {"entry": self.entries[i], "similarity": float(sims[i])}
                for i in top_indices
                if i < len(self.entries) and sims[i] > 0
            ]
        except Exception as e:
            logger.error(f"TF-IDF search error: {e}")
            return self._search_keyword(query, top_k)

    def _search_keyword(self, query: str, top_k: int) -> List[Dict]:
        """Last-resort: keyword overlap similarity."""
        q_words = set(query.lower().split())
        scored = []
        for entry in self.entries:
            e_words = set(entry["question"].lower().split())
            overlap = len(q_words & e_words)
            if overlap:
                sim = overlap / max(len(q_words), len(e_words))
                scored.append({"entry": entry, "similarity": sim})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]
