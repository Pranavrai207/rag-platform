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
from database import engine, Base, get_db
from config import config

# Initialize DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Multi-Tenant RAG Platform")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# ─── Auth Routes ────────────────────────────────────────

@app.post("/auth/register")
async def register(
    tenant_id: str = Form(...),
    tenant_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        print(f"Registration attempt: tenant={tenant_id}, email={email}")
        
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
async def login(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not auth.verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(
        data={"sub": user.email, "tenant_id": user.tenant_id}
    )
    return {"access_token": access_token, "token_type": "bearer", "tenant_id": user.tenant_id}

# ─── Document Routes ────────────────────────────────────

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(config.UPLOAD_DIR, f"{current_user.tenant_id}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    db_doc = models.Document(
        filename=file.filename,
        file_type=file.filename.split('.')[-1],
        tenant_id=current_user.tenant_id
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    
    try:
        await ingestion.process_document(file_path, current_user.tenant_id, db_doc.id)
    except Exception as e:
        db.delete(db_doc)
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"message": "File uploaded and processed", "id": db_doc.id}

@app.get("/documents")
async def list_documents(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    docs = db.query(models.Document).filter(models.Document.tenant_id == current_user.tenant_id).all()
    return [{"id": d.id, "filename": d.filename, "file_type": d.file_type, "created_at": str(d.created_at)} for d in docs]

@app.delete("/documents/{doc_id}")
async def delete_document(
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
    return {"message": "Document deleted"}

# ─── Chat Routes ────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(
    question: str = Form(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    return await chat.chat_streaming_response(current_user.tenant_id, current_user.id, question, db)

@app.get("/chat/history")
async def chat_history(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    history = db.query(models.ChatMessage).filter(
        models.ChatMessage.tenant_id == current_user.tenant_id,
        models.ChatMessage.user_id == current_user.id
    ).order_by(models.ChatMessage.created_at.desc()).limit(20).all()
    
    return [{"role": m.role, "content": m.content, "created_at": str(m.created_at)} for m in reversed(history)]

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
