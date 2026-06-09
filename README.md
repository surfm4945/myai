# AI Chat — Local AI Chat Application

A professional, ChatGPT-style AI chat interface that runs **100% locally** — no API keys, no internet connection required.

## Features

- 🤖 **Local AI** — powered by DialoGPT (Microsoft) or intelligent template-based generation
- 🧠 **Persistent Memory** — `brain.kesar` stores learned responses and grows over time
- 🔍 **Semantic Search** — TF-IDF + cosine similarity (scikit-learn) finds relevant stored knowledge
- 💬 **Session Management** — save, load, and manage multiple chat sessions
- 🎓 **Teach Mode** — add custom Q&A pairs directly via the sidebar
- 🌙 **Dark Mode** — clean, modern ChatGPT-style interface
- ⚡ **Instant Startup** — no model download required on first run (uses template mode until torch model loads)

## Project Structure

```
ai-chat-app/
├── app.py                   # Main Streamlit application
├── model/
│   ├── loader.py            # Local model loading (DialoGPT + fallback)
│   ├── trainer.py           # brain.kesar management
│   └── brain.kesar          # AI persistent memory (JSON format)
├── utils/
│   ├── memory.py            # Chat session management
│   └── embeddings.py        # TF-IDF / FAISS semantic search
├── data/
│   └── knowledge.json       # Supplementary knowledge base
├── sessions/                # Per-chat session files (auto-created)
├── .streamlit/
│   └── config.toml          # Dark theme configuration
└── requirements.txt
```

## How to Run

```bash
streamlit run app.py --server.port 5000
```

## Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| LLM | DialoGPT-small (HuggingFace Transformers) |
| Semantic Search | scikit-learn TF-IDF + cosine similarity |
| Memory | JSON (brain.kesar) + file-based sessions |
| Fallback | Template-based response generation |

## NO API KEY REQUIRED

All AI processing happens locally on your machine:
- The DialoGPT model downloads from HuggingFace once and caches locally
- All conversations are stored in `sessions/` as JSON files
- The brain.kesar file grows as you chat — the AI learns from every conversation

## Adding Custom Knowledge

Use the **🎓 Teach the AI** panel in the sidebar to add custom Q&A pairs directly to `brain.kesar`. The AI will use this knowledge in future conversations.

## Architecture

1. **User sends message** → stored in session file
2. **Semantic search** → TF-IDF finds similar brain.kesar entries  
3. **High match (≥82%)** → brain answer returned (optionally enhanced by model)
4. **Low/no match** → DialoGPT generates novel response
5. **Exchange saved** → added to brain.kesar for future retrieval

## Future Extensions

- Voice input/output (Web Speech API or whisper.cpp)
- File upload and document Q&A
- Plugin system for custom tools
- Upgrade to LLaMA/Mistral for better quality
- FAISS vector database for large-scale memory
"# Ai_Chat" 
