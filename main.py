import os
import shutil
import traceback
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List

import models
import auth
import ingestion
import chat
import asyncio
import threading
from datetime import datetime
from database import engine, Base, get_db, SessionLocal
from config import config
import security
from security import limiter, log_event, sanitize_text, validate_filename, validate_file_mime, detect_prompt_injection
import data_update
from admin import admin_router
import task_queue as tq

# Initialize DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Multi-Tenant RAG Platform")

# Security middleware
app.middleware("http")(security.security_headers_middleware)

# Rate Limiting
app.state.limiter = limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Admin Router
app.include_router(admin_router)

# CORS middleware
origins = config.ALLOWED_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"\n--- UNHANDLED ERROR ---")
    print(f"Path: {request.url.path}")
    print(f"Exception: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    print(f"-----------------------\n")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

@app.on_event("startup")
async def startup_event():
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(config.OLLAMA_BASE_URL)
            if response.status_code == 200:
                print("✓ Ollama is running")
            else:
                print("⚠ Ollama not running. Start it with: ollama serve")
    except Exception:
        print("⚠ Ollama not running. Start it with: ollama serve")
    
    # Start background ingestion worker
    asyncio.create_task(tq.ingestion_worker(SessionLocal))
    print("✓ Background ingestion queue worker started")
    
    # Start automation threads
    os.makedirs(config.WATCHED_FOLDERS, exist_ok=True)
    data_update.start_folder_watcher()
    data_update.start_scheduler()

# ─── Auth Routes ────────────────────────────────────────

@app.post("/auth/register")
@limiter.limit(f"{config.RATE_LIMIT_UPLOAD}/minute") # Using upload limit for registration
async def register(
    request: Request,
    tenant_id: str = Form(...),
    tenant_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        # Sanitize inputs
        tenant_id = sanitize_text(tenant_id)
        tenant_name = sanitize_text(tenant_name)
        email = sanitize_text(email)
        
        # Check if tenant exists
        db_tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
        if not db_tenant:
            db_tenant = models.Tenant(id=tenant_id, name=tenant_name)
            db.add(db_tenant)
        
        # Check if user exists
        db_user = db.query(models.User).filter(models.User.email == email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed = auth.get_password_hash(password)
        new_user = models.User(
            email=email,
            hashed_password=hashed,
            tenant_id=tenant_id,
            role="admin"
        )
        db.add(new_user)
        db.commit()
        print(f"✓ User {email} registered successfully")
        return {"message": "Registration successful"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Registration error: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login")
@limiter.limit("5/minute")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    email = sanitize_text(email)
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if user:
        # Check lockout
        security.check_account_locked(user)
        
    if not user or not auth.verify_password(password, user.hashed_password):
        if user:
            security.record_failed_attempt(db, user)
        log_event("LOGIN_FAILED", user.tenant_id if user else "unknown", user.id if user else "unknown", request, 401, detail=f"Failed login for {email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Reset failed attempts on success
    security.reset_failed_attempts(db, user)
    log_event("LOGIN", user.tenant_id, user.id, request, 200)
    
    access_token = auth.create_access_token(
        data={
            "sub": user.email, 
            "tenant_id": user.tenant_id,
            "role": user.role # Include role for RBAC
        }
    )
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "tenant_id": user.tenant_id,
        "role": user.role
    }

# ─── Document Routes ────────────────────────────────────

@app.post("/documents/upload")
@limiter.limit(f"{config.RATE_LIMIT_UPLOAD}/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    # Validate filename
    validate_filename(file.filename)
    
    # Validate file mime type/magic bytes
    content = await file.read()
    validate_file_mime(content, file.filename)
    await file.seek(0) # Reset file pointer for shutil
    
    file_path = os.path.join(config.UPLOAD_DIR, f"{current_user.tenant_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Compute checksum
    checksum = ingestion.compute_checksum(file_path)
    
    # Create job record first (so we can return instantly)
    job = models.IngestionJob(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        filename=file.filename,
        status="queued"
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Create document record
    db_doc = models.Document(
        filename=file.filename,
        file_type=file.filename.split('.')[-1],
        tenant_id=current_user.tenant_id,
        checksum=checksum,
        version=1
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    
    # Link job to document
    job.doc_id = db_doc.id
    db.commit()

    # Record first version (chunk_count updated after processing)
    version_record = models.DocumentVersion(
        doc_id=db_doc.id,
        version_number=1,
        checksum=checksum,
        chunk_count=0
    )
    db.add(version_record)
    db.commit()
    db.refresh(version_record)

    # Enqueue — returns immediately, worker runs in background
    await tq.enqueue_ingestion(
        job_id=job.id,
        file_path=file_path,
        tenant_id=current_user.tenant_id,
        doc_id=db_doc.id,
        version_record_id=version_record.id,
        version=1
    )

    log_event("UPLOAD_QUEUED", current_user.tenant_id, current_user.id, request, 202, detail=f"Queued {file.filename}")
    return {
        "message": "File queued for processing",
        "job_id": job.id,
        "doc_id": db_doc.id,
        "status": "queued"
    }

@app.get("/documents")
@limiter.limit(f"{config.RATE_LIMIT_UPLOAD}/minute")
async def list_documents(
    request: Request,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    docs = db.query(models.Document).filter(models.Document.tenant_id == current_user.tenant_id).all()
    return [{"id": d.id, "filename": d.filename, "file_type": d.file_type, "created_at": str(d.created_at)} for d in docs]

@app.delete("/documents/{doc_id}")
@limiter.limit(f"{config.RATE_LIMIT_UPLOAD}/minute")
async def delete_document(
    request: Request,
    doc_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    doc = db.query(models.Document).filter(
        models.Document.id == doc_id, 
        models.Document.tenant_id == current_user.tenant_id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    await ingestion.delete_document_from_vector_store(current_user.tenant_id, doc.id)
    
    file_path = os.path.join(config.UPLOAD_DIR, f"{current_user.tenant_id}_{doc.filename}")
    if os.path.exists(file_path):
        os.remove(file_path)
    
    db.delete(doc)
    db.commit()
    log_event("DELETE", current_user.tenant_id, current_user.id, request, 200, detail=f"Deleted doc {doc_id}")
    return {"message": "Document deleted"}

@app.get("/documents/jobs")
@limiter.limit(f"{config.RATE_LIMIT_UPLOAD}/minute")
async def list_ingestion_jobs(
    request: Request,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """List recent ingestion jobs for this tenant."""
    jobs = db.query(models.IngestionJob).filter(
        models.IngestionJob.tenant_id == current_user.tenant_id
    ).order_by(models.IngestionJob.created_at.desc()).limit(50).all()
    
    return [{
        "id": j.id, "doc_id": j.doc_id, "filename": j.filename,
        "status": j.status, "progress_pct": j.progress_pct,
        "error_message": j.error_message, "created_at": str(j.created_at),
        "completed_at": str(j.completed_at) if j.completed_at else None
    } for j in jobs]

@app.get("/documents/jobs/{job_id}")
@limiter.limit(f"{config.RATE_LIMIT_UPLOAD}/minute")
async def get_ingestion_job(
    request: Request,
    job_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """Get status of a specific ingestion job."""
    job = db.query(models.IngestionJob).filter(
        models.IngestionJob.id == job_id,
        models.IngestionJob.tenant_id == current_user.tenant_id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "id": job.id, "doc_id": job.doc_id, "filename": job.filename,
        "status": job.status, "progress_pct": job.progress_pct,
        "error_message": job.error_message, "created_at": str(job.created_at),
        "completed_at": str(job.completed_at) if job.completed_at else None
    }



# ─── Chat Routes ────────────────────────────────────────

@app.post("/chat")
@limiter.limit(f"{config.RATE_LIMIT_CHAT}/minute")
async def chat_endpoint(
    request: Request,
    question: str = Form(...),
    session_id: int = Form(None),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    # Sanitize and detect injection
    question = sanitize_text(question)
    if detect_prompt_injection(question):
        log_event("BLOCKED_INJECTION", current_user.tenant_id, current_user.id, request, 400, detail=f"Injection detected in: {question}")
        raise HTTPException(status_code=400, detail={"error": "invalid_query", "detail": "Query contains disallowed patterns"})
    
    log_event("CHAT", current_user.tenant_id, current_user.id, request, 200)
    return await chat.chat_streaming_response(current_user.tenant_id, current_user.id, question, session_id, db)

@app.get("/chat/sessions")
@limiter.limit(f"{config.RATE_LIMIT_CHAT}/minute")
async def get_chat_sessions(
    request: Request,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    sessions = db.query(models.ChatSession).filter(
        models.ChatSession.tenant_id == current_user.tenant_id,
        models.ChatSession.user_id == current_user.id
    ).order_by(models.ChatSession.created_at.desc()).all()
    
    return [{"id": s.id, "title": s.title, "created_at": str(s.created_at)} for s in sessions]

@app.put("/chat/sessions/{session_id}")
@limiter.limit(f"{config.RATE_LIMIT_CHAT}/minute")
async def rename_chat_session(
    request: Request,
    session_id: int,
    title: str = Form(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.tenant_id == current_user.tenant_id,
        models.ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session.title = sanitize_text(title)
    db.commit()
    return {"message": "Session renamed successfully"}

@app.delete("/chat/sessions/{session_id}")
@limiter.limit(f"{config.RATE_LIMIT_CHAT}/minute")
async def delete_chat_session(
    request: Request,
    session_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.tenant_id == current_user.tenant_id,
        models.ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    db.delete(session) # Messages deleted via cascade
    db.commit()
    return {"message": "Session deleted"}

@app.get("/chat/history")
@limiter.limit(f"{config.RATE_LIMIT_CHAT}/minute")
async def chat_history(
    request: Request,
    session_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    history = db.query(models.ChatMessage).filter(
        models.ChatMessage.tenant_id == current_user.tenant_id,
        models.ChatMessage.user_id == current_user.id,
        models.ChatMessage.session_id == session_id
    ).order_by(models.ChatMessage.created_at.asc()).all()
    
    return [{"role": m.role, "content": m.content, "created_at": str(m.created_at)} for m in history]

# ─── Data Update Routes ─────────────────────────────────

@app.post("/data/reingest/{doc_id}")
@limiter.limit("5/minute")
async def reingest_endpoint(
    request: Request,
    doc_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    doc = db.query(models.Document).filter(
        models.Document.id == doc_id,
        models.Document.tenant_id == current_user.tenant_id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    file_path = os.path.join(config.UPLOAD_DIR, f"{current_user.tenant_id}_{doc.filename}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Original file not found in uploads")
        
    result = await data_update.reingest_document(doc_id, current_user.tenant_id, file_path, db)
    return result

@app.get("/documents/{doc_id}/versions")
async def get_versions(
    doc_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    doc = db.query(models.Document).filter(
        models.Document.id == doc_id,
        models.Document.tenant_id == current_user.tenant_id
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    versions = db.query(models.DocumentVersion).filter(
        models.DocumentVersion.doc_id == doc_id
    ).order_by(models.DocumentVersion.version_number.desc()).all()
    
    return [
        {
            "version_number": v.version_number,
            "checksum": v.checksum,
            "chunk_count": v.chunk_count,
            "created_at": str(v.created_at)
        } for v in versions
    ]

@app.post("/documents/{doc_id}/rollback/{version}")
async def rollback_endpoint(
    doc_id: int,
    version: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    result = data_update.rollback_document(doc_id, version, current_user.tenant_id, db)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/data/sync")
async def manual_sync(
    current_user: models.User = Depends(auth.get_current_user)
):
    # Triggers background sync
    threading.Thread(target=data_update.run_scheduled_sync, daemon=True).start()
    return {"status": "sync_started"}

@app.get("/data/status")
async def data_status(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    total_docs = db.query(models.Document).filter(models.Document.tenant_id == current_user.tenant_id).count()
    
    # In a real app we'd track watched files more accurately
    watched_dir = os.path.join(config.WATCHED_FOLDERS, current_user.tenant_id)
    watched_count = 0
    if os.path.exists(watched_dir):
        watched_count = len([f for f in os.listdir(watched_dir) if os.path.isfile(os.path.join(watched_dir, f))])
        
    return {
        "watched_files": watched_count,
        "last_sync": str(datetime.utcnow()), # Placeholder
        "pending_ingestions": 0,
        "total_documents": total_docs
    }

@app.post("/data/webhook")
async def data_webhook(
    request: Request,
    db: Session = Depends(get_db)
):
    webhook_secret = request.headers.get("X-Webhook-Secret")
    if not webhook_secret or not hmac.compare_digest(webhook_secret, config.WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
        
    data = await request.json()
    tenant_id = data.get("tenant_id")
    filename = data.get("filename")
    action = data.get("action")
    
    if not all([tenant_id, filename, action]):
        raise HTTPException(status_code=400, detail="Missing required fields")
        
    if action in ["add", "update"]:
        # Find document
        doc = db.query(models.Document).filter(
            models.Document.filename == filename,
            models.Document.tenant_id == tenant_id
        ).first()
        if doc:
            file_path = os.path.join(config.UPLOAD_DIR, f"{tenant_id}_{filename}")
            if os.path.exists(file_path):
                # Trigger re-ingestion
                import asyncio
                threading.Thread(target=lambda: asyncio.run(data_update.reingest_document(doc.id, tenant_id, file_path, SessionLocal())), daemon=True).start()
    
    return {"status": "queued", "job_id": "auto-generated-id"}

# ─── Static Files ───────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Create dirs
os.makedirs("static", exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting RAG Platform on http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
