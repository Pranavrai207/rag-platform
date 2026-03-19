# Multi-Tenant RAG Platform 🚀

A production-ready, multi-tenant Retrieval-Augmented Generation (RAG) platform that runs **100% locally and offline**.

## ✨ Features

- **Multi-Tenancy**: Granular data isolation for different tenants.
- **Local RAG**: Powered by **Ollama** (`gemma3:4b` / `nomic-embed-text`) and **ChromaDB**.
- **Secure Auth**: JWT-based authentication with direct `bcrypt` hashing.
- **Real-time Chat**: Streaming responses via SSE with source citations.
- **Multi-Format Ingestion**: Supports PDF, DOCX, CSV, and TXT files.
- **Zero-Dependency Frontend**: Single `index.html` using Vanilla JS and Tailwind CSS CDN.

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, SQLAlchemy (SQLite)
- **RAG**: LangChain, Ollama, ChromaDB
- **Auth**: python-jose, bcrypt
- **Frontend**: Vanilla JS, Tailwind CSS

## 🚀 Quick Start

### 1. Prerequisites
Install [Ollama](https://ollama.com/) and pull the required models:
```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

### 2. Setup
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file based on `.env.example`:
```env
JWT_SECRET_KEY=your-random-secret-key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=gemma3:4b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 4. Run
```bash
python main.py
```
Visit `http://localhost:8000` to register your first tenant!

## 🔒 Security Note
This project is designed for local-first deployment. Ensure `JWT_SECRET_KEY` is kept private and not committed to version control.