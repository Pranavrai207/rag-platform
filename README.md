# RagPlatform 🚀
### Industrial-Grade, Local-First Multi-Tenant RAG

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Offline-orange)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A performance-optimized, security-hardened, multi-tenant Retrieval-Augmented Generation (RAG) platform designed to run **100% locally and offline**.

---

## 💎 Advanced Features (v2.0)

- **⚡ Async Background Ingestion**: Non-blocking document processing using an `asyncio.Queue` worker. Upload gigabytes of data while continuing to chat; track real-time status with live badges.
- **💬 Chat Sessions & History**: Full conversational memory management with isolated context threads, automatic session titling, and a persistent sidebar history.
- **🎨 Minimalist UI Redesign**: A premium, "Light-Mode" aesthetic featuring **Google Sans** typography, a Solar White background, and striking Orange/Blue accents.
- **🛡️ Industrial Security**: JWT-based tenant isolation, Rate Limiting, Input Sanitization, Prompt Injection Detection, and a comprehensive Audit Trail.
- **🛠️ Admin Command Center**: Real-time telemetry, storage monitoring, and live LLM model orchestration for superadmins.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.10+), SQLAlchemy (SQLite)
- **RAG Engine**: LangChain, ChromaDB (Vector Store), semantic-chunking
- **AI Models**: Ollama (`gemma3:4b` LLM, `nomic-embed-text` Embeddings)
- **Frontend**: Single-Page App (Vanilla JS, Tailwind CSS, Google Sans / Outfit)

---

## 🚀 Quick Start

### 1. Model Provisioning
Ensure [Ollama](https://ollama.com/) is running locally and pull the specialized models:
```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

### 2. Implementation
```bash
# Clone and install
pip install -r requirements.txt

# Run the platform
python main.py
```

### 3. Configuration
Configure your security tokens in `.env`:
```env
JWT_SECRET_KEY=your_secure_random_key_here
OLLAMA_BASE_URL=http://localhost:11434
UPLOAD_DIR=./uploads
```

---

## 🔐 Multi-Tenancy Design
Data is strictly isolated at the database and vector-store level using `tenant_id` filters. Each tenant manages their own users, documents, and chat histories without any cross-border data leakage.

---

## 🔒 Security Note
Designed for enterprise/local privacy. The platform respects air-gapped constraints and never sends data to external APIs. Always keep your `JWT_SECRET_KEY` private.