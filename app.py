"""
app.py - AI Chat Application
A professional, ChatGPT-style AI chat interface powered by local AI.
No API keys required. All processing runs locally.
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Add project root to path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)

from model.loader import ModelLoader
from model.trainer import BrainTrainer
from utils.memory import ChatMemory
from utils.embeddings import EmbeddingSearch

# ── Custom CSS — dark ChatGPT-style theme ──────────────────────────────────────
st.markdown(
    """
<style>
/* ── Reset & base ─────────────────────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ── App background ─────────────────────────────────────────────────────────  */
.stApp {
    background-color: #343541;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────────  */
[data-testid="stSidebar"] {
    background-color: #202123 !important;
    border-right: 1px solid #3e3f4b;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label {
    color: #c5c5d2 !important;
    font-size: 0.875rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ececf1 !important;
}

/* ── Sidebar buttons (chat sessions) ────────────────────────────────────────  */
[data-testid="stSidebar"] .stButton button {
    background-color: transparent !important;
    color: #c5c5d2 !important;
    border: none !important;
    border-radius: 6px !important;
    text-align: left !important;
    padding: 8px 10px !important;
    font-size: 0.82rem !important;
    width: 100% !important;
    transition: background-color 0.15s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #2a2b32 !important;
    color: #ececf1 !important;
}

/* ── Primary action buttons ──────────────────────────────────────────────────  */
.primary-btn button,
[data-testid="stSidebar"] .primary-btn button {
    background-color: #10a37f !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.primary-btn button:hover {
    background-color: #0d8f6b !important;
}

/* ── Main chat area ──────────────────────────────────────────────────────────  */
.main .block-container {
    max-width: 820px;
    padding: 2rem 1.5rem 6rem;
}

/* ── Chat messages ───────────────────────────────────────────────────────────  */
[data-testid="stChatMessage"] {
    border-radius: 0 !important;
    padding: 18px 0 !important;
}
[data-testid="stChatMessage"][data-role="user"] {
    background-color: #343541 !important;
}
[data-testid="stChatMessage"][data-role="assistant"] {
    background-color: #444654 !important;
    margin: 0 -1.5rem;
    padding: 18px 1.5rem !important;
}

/* ── Message text ────────────────────────────────────────────────────────────  */
[data-testid="stChatMessage"] .stMarkdown p,
[data-testid="stChatMessage"] .stMarkdown li,
[data-testid="stChatMessage"] .stMarkdown code {
    color: #ececf1 !important;
    line-height: 1.7 !important;
}
[data-testid="stChatMessage"] .stMarkdown code {
    background-color: #2d2d3b !important;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.9em;
}
[data-testid="stChatMessage"] .stMarkdown pre {
    background-color: #1a1a2e !important;
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid #3e3f4b;
}

/* ── Page title / subtitle ───────────────────────────────────────────────────  */
h1 { color: #ececf1 !important; }
h2, h3 { color: #d1d5db !important; }
p, li { color: #c5c5d2 !important; }
.stMarkdown p { color: #ececf1 !important; }

/* ── Chat input ──────────────────────────────────────────────────────────────  */
[data-testid="stChatInput"] {
    background-color: #40414f !important;
    border-radius: 12px !important;
    border: 1px solid #565869 !important;
}
[data-testid="stChatInput"] textarea {
    color: #ececf1 !important;
    background-color: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #8e8ea0 !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────────────  */
[data-testid="stMetric"] label { color: #8e8ea0 !important; font-size: 0.75rem; }
[data-testid="stMetricValue"] { color: #10a37f !important; font-weight: 700; }

/* ── Expander ────────────────────────────────────────────────────────────────  */
[data-testid="stExpander"] {
    background-color: #2a2b32 !important;
    border: 1px solid #3e3f4b !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: #c5c5d2 !important; }
[data-testid="stExpander"] .stTextInput input,
[data-testid="stExpander"] .stTextArea textarea {
    background-color: #343541 !important;
    color: #ececf1 !important;
    border: 1px solid #565869 !important;
    border-radius: 6px !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────────────  */
.stSpinner > div { border-top-color: #10a37f !important; }

/* ── Dividers ────────────────────────────────────────────────────────────────  */
hr { border-color: #3e3f4b !important; }

/* ── Alert / info boxes ──────────────────────────────────────────────────────  */
.stAlert { border-radius: 8px !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Cached resource loaders ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model() -> ModelLoader:
    loader = ModelLoader()
    loader.load()
    return loader


@st.cache_resource(show_spinner=False)
def get_search() -> EmbeddingSearch:
    brain_path = PROJECT_ROOT / "model" / "brain.kesar"
    search = EmbeddingSearch(str(PROJECT_ROOT / "model"))
    search.load_from_brain(str(brain_path))
    return search


def get_memory() -> ChatMemory:
    return ChatMemory(str(PROJECT_ROOT / "sessions"))


def get_trainer() -> BrainTrainer:
    return BrainTrainer(str(PROJECT_ROOT / "model" / "brain.kesar"))


# ── Session state init ─────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_status" not in st.session_state:
    st.session_state.model_status = "loading"


def _start_new_session():
    memory = get_memory()
    sid = memory.create_session()
    st.session_state.session_id = sid
    st.session_state.messages = []


def _load_session(session_id: str):
    memory = get_memory()
    msgs = memory.get_messages(session_id)
    st.session_state.session_id = session_id
    st.session_state.messages = msgs


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 AI Chat")
    st.caption("Local AI • No API keys required")
    st.divider()

    # New chat
    col_new, _ = st.columns([3, 1])
    with col_new:
        if st.button("✦  New Chat", use_container_width=True, key="new_chat_btn"):
            _start_new_session()
            st.rerun()

    st.divider()
    st.markdown("**Recent Chats**")

    memory = get_memory()
    sessions = memory.list_sessions()

    if sessions:
        for sess in sessions[:20]:
            is_active = sess["id"] == st.session_state.session_id
            title = sess.get("title", "New Chat")
            if len(title) > 22:
                title = title[:22] + "…"
            active_dot = "🟢 " if is_active else "💬 "

            c1, c2, c3 = st.columns([1, 5, 1])
            with c1:
                if st.button("📂", key=f"o_{sess['id']}", help="Open this chat"):
                    _load_session(sess["id"])
                    st.rerun()
            with c2:
                display = f"**{title}**" if is_active else title
                st.markdown(
                    f"<div style='padding:6px 0 0;color:#ececf1;font-size:0.83rem;'>"
                    f"{active_dot}{display}</div>",
                    unsafe_allow_html=True,
                )
            with c3:
                if st.button("🗑", key=f"d_{sess['id']}", help="Delete this chat"):
                    memory.delete_session(sess["id"])
                    if st.session_state.session_id == sess["id"]:
                        st.session_state.session_id = None
                        st.session_state.messages = []
                    st.rerun()
    else:
        st.caption("No chats yet. Start one above!")

    st.divider()

    # Brain stats
    trainer = get_trainer()
    brain = trainer.get_brain_data()
    stats = brain.get("stats", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🧠 Entries", stats.get("total_entries", 0))
    with c2:
        st.metric("💬 Chats", stats.get("total_conversations", 0))
    with c3:
        st.metric("📨 Msgs", stats.get("total_messages", 0))

    st.divider()

    # Teach the AI
    with st.expander("🎓  Teach the AI"):
        st.caption("Add custom knowledge to the brain.kesar file")
        teach_q = st.text_input("Question", placeholder="What is quantum computing?", key="teach_q")
        teach_a = st.text_area("Answer", placeholder="Quantum computing uses...", key="teach_a", height=90)
        if st.button("💾  Save to Brain", key="save_brain", use_container_width=True):
            if teach_q.strip() and teach_a.strip():
                t = get_trainer()
                eid = t.add_entry(teach_q.strip(), teach_a.strip(), tags=["taught"])
                s = get_search()
                s.add_entry(teach_q.strip(), teach_a.strip(), eid)
                st.success("✅ Knowledge saved to brain!")
                time.sleep(0.8)
                st.rerun()
            else:
                st.warning("Please fill in both fields.")

    st.divider()

    # Model status
    model = get_model()
    if model.is_neural():
        st.success("🟢 DialoGPT model active")
    else:
        st.info("🟡 Template mode — Model loading")

    # Clear all
    if st.button("🗑️  Clear All Chats", use_container_width=True, key="clear_all"):
        memory.clear_all()
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()


# ── Main content ───────────────────────────────────────────────────────────────

# Ensure a session exists
if st.session_state.session_id is None:
    _start_new_session()

# Header
st.markdown("# AI Chat")
st.caption("Powered by local AI — all processing happens on your machine")
st.divider()

# ── Render message history ─────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown(
        """
<div style="text-align:center; padding:48px 0 32px; color:#8e8ea0;">
  <div style="font-size:3rem; margin-bottom:12px;">🤖</div>
  <h3 style="color:#ececf1; margin-bottom:8px;">How can I help you today?</h3>
  <p style="color:#8e8ea0; font-size:0.9rem;">Ask me anything. I learn from every conversation.</p>
  <p style="color:#6e6e80; font-size:0.8rem; margin-top:16px;">
    💡 Tip: Use the <strong>🎓 Teach the AI</strong> panel to add your own knowledge.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────

user_input = st.chat_input("Send a message…")

if user_input and user_input.strip():
    text = user_input.strip()
    memory = get_memory()
    trainer = get_trainer()
    search = get_search()
    model = get_model()

    # ── Render user message ────────────────────────────────────────────────────
    with st.chat_message("user"):
        st.markdown(text)

    # Persist user message
    memory.add_message(st.session_state.session_id, "user", text)
    st.session_state.messages.append({"role": "user", "content": text})

    # ── Generate response ──────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = ""

            # 1. Semantic search in brain
            similar = search.search(text, top_k=3)

            high_match   = similar and similar[0]["similarity"] >= 0.55
            medium_match = similar and similar[0]["similarity"] >= 0.35

            if high_match:
                # Brain answer is authoritative — return it directly (clean, curated)
                response = similar[0]["entry"]["answer"]
            else:
                # Generate with model, then quality-check
                raw = model.generate(
                    text,
                    history=st.session_state.messages[:-1],
                )

                if model.is_quality(raw):
                    response = raw
                elif medium_match:
                    # Neural output is bad — fall back to partial brain match
                    response = similar[0]["entry"]["answer"]
                else:
                    # Neural output is bad, no brain match — use clean template
                    response = model.safe_template(text)

            # Final safety net
            if not response or not response.strip():
                response = "I'm not sure how to answer that. Could you rephrase or give me more context? You can also teach me the answer using the **🎓 Teach the AI** panel in the sidebar."

        st.markdown(response)

    # ── Persist response ───────────────────────────────────────────────────────
    memory.add_message(st.session_state.session_id, "assistant", response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Update brain with this exchange
    try:
        eid = trainer.add_entry(text, response)
        search.add_entry(text, response, eid or f"conv_{int(time.time())}")
    except Exception:
        pass

    trainer.increment_stats()

    # Auto-title the session from the first user message
    sessions = memory.list_sessions()
    for sess in sessions:
        if sess["id"] == st.session_state.session_id:
            if sess.get("title") in ("New Chat", "", None):
                memory.update_session_title(st.session_state.session_id, text[:50])
            break
