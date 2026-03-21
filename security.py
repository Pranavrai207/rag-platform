import os
import re
import json
import logging
import unicodedata
import bleach
import asyncio
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from fastapi import Request, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from jose import jwt, JWTError
from config import config
import models

# ─── Audit Logging ──────────────────────────────────────

# Ensure log directory exists
os.makedirs(config.LOG_DIR, exist_ok=True)

audit_logger = logging.getLogger("audit_logger")
audit_logger.setLevel(logging.INFO)
audit_handler = RotatingFileHandler(
    os.path.join(config.LOG_DIR, "audit.log"),
    maxBytes=10*1024*1024, # 10MB
    backupCount=5
)
audit_logger.addHandler(audit_handler)

def log_event(event_type, tenant_id, user_id, request: Request, status_code, detail=""):
    """Writes an audit log entry in JSON Lines format (asynchronously)."""
    async def _log():
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "ip_address": request.client.host if request.client and request.client.host else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
                "status": status_code,
                "detail": detail
            }
            # Standard logging is blocking, so we run it in a separate thread if possible
            # But since we are calling this from sync or async, it's tricky.
            # In FastAPI, we can use background tasks or just run_in_executor if in async.
            audit_logger.info(json.dumps(log_entry))
        except Exception:
            pass

    # Fire and forget
    asyncio.create_task(_log())

# ─── Rate Limiting ──────────────────────────────────────

def get_tenant_id_from_token(request: Request):
    """Extract tenant_id from JWT token in Authorization header."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.ALGORITHM])
            return payload.get("tenant_id")
        except JWTError:
            pass
    return get_remote_address(request) # Fallback to IP

limiter = Limiter(key_func=get_tenant_id_from_token)

async def rate_limit_exceeded_custom_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for rate limit exceeded."""
    retry_after = exc.detail.split("at ")[-1] if "at " in exc.detail else "60"
    # log_event("RATE_LIMITED", "unknown", "unknown", request, 429, detail=f"Limit exceeded: {exc.detail}")
    return HTTPException(
        status_code=429,
        detail={"error": "rate_limit_exceeded", "retry_after_seconds": retry_after}
    )

# ─── Input Sanitization ──────────────────────────────────

def sanitize_text(text: str, max_length: int = None) -> str:
    """Strip HTML, normalize unicode, and enforce max length."""
    if not text:
        return ""
    
    # Enforce max length
    limit = max_length or config.MAX_QUERY_LENGTH
    text = text[:limit]
    
    # Strip HTML tags
    text = bleach.clean(text, tags=[], strip=True)
    
    # Remove null bytes and control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    
    return text.strip()

def validate_filename(filename: str) -> bool:
    """Validate filename for path traversal and allowed characters."""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename missing")
    
    if len(filename) > 255:
        raise HTTPException(status_code=400, detail="Filename too long")
        
    if ".." in filename or filename.startswith("/") or "\\" in filename:
        raise HTTPException(status_code=400, detail="Path traversal detected in filename")
        
    if "\0" in filename:
        raise HTTPException(status_code=400, detail="Null byte in filename")
        
    # Only allow alphanumeric, dots, dashes, underscores, spaces
    if not re.match(r"^[a-zA-Z0-9._\s-]+$", filename):
        raise HTTPException(status_code=400, detail="Invalid characters in filename")
        
    return True

def validate_file_mime(file_bytes: bytes, filename: str) -> bool:
    """Check magic bytes for allowed types: PDF, DOCX/ZIP, TXT, CSV."""
    if file_bytes.startswith(b"%PDF"):
        return True
    
    if file_bytes.startswith(b"PK\x03\x04"): # ZIP header (used by DOCX)
        return True
        
    # For TXT and CSV, check if valid UTF-8
    try:
        file_bytes.decode("utf-8")
        return True
    except UnicodeDecodeError:
        pass
        
    raise HTTPException(status_code=415, detail="Unsupported file type (invalid magic bytes or encoding)")

# ─── Prompt Injection Detection ──────────────────────────

INJECTION_PATTERNS = {
    "role_override": [
        "ignore previous", "forget instructions", "you are now", 
        "act as", "pretend you are", "new persona"
    ],
    "jailbreak": [
        "DAN", "do anything now", "developer mode", 
        "jailbreak", "no restrictions", "bypass"
    ],
    "instruction_injection": [
        "system prompt", "your instructions", "disregard", 
        "override", "new task:"
    ]
}

def detect_prompt_injection(query: str) -> bool:
    """Check for prompt injection patterns."""
    query_lower = query.lower()
    for category, patterns in INJECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in query_lower:
                return True
    return False

# ─── Security Headers Middleware ─────────────────────────

async def security_headers_middleware(request: Request, call_next):
    """Add security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' cdn.tailwindcss.com cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' cdn.tailwindcss.com fonts.googleapis.com; "
        "font-src 'self' fonts.gstatic.com; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
    )
    response.headers["Content-Security-Policy"] = csp
    return response

# ─── Failed Login Lockout ────────────────────────────────

def check_account_locked(user: models.User):
    """Check if account is currently locked."""
    if user.locked_until and user.locked_until > datetime.utcnow():
        retry_after = int((user.locked_until - datetime.utcnow()).total_seconds())
        raise HTTPException(
            status_code=423, # Locked
            detail={"error": "account_locked", "retry_after_seconds": retry_after}
        )

def record_failed_attempt(db, user: models.User):
    """Increment failed attempts and lock if needed."""
    user.failed_login_attempts += 1
    if user.failed_login_attempts >= 5:
        user.locked_until = datetime.utcnow() + timedelta(minutes=15)
        # log_event("ADMIN_ACTION", user.tenant_id, user.id, None, 423, detail="Account locked due to failed attempts")
    db.commit()

def reset_failed_attempts(db, user: models.User):
    """Clear failed attempts on successful login."""
    user.failed_login_attempts = 0
    user.locked_until = None
    db.commit()
