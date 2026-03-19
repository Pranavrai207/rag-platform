import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-it")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma3:4b")
    OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    DATABASE_URL = "sqlite:///./rag_platform.db"

config = Config()

# Ensure directories exist
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
