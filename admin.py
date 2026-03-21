import os
import json
import logging
import shutil
import httpx
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session
from sqlalchemy import func

import models
import auth
from database import get_db, SessionLocal
from config import config
from security import log_event
import ingestion

# Role hierarchy
ROLES = {
    "superadmin": 4,
    "admin": 3,
    "member": 2,
    "readonly": 1
}

# Configure logging
logger = logging.getLogger(__name__)

admin_router = APIRouter(prefix="/admin", tags=["admin"])

# ─── RBAC Enforcement ───────────────────────────────────

def require_role(*allowed_roles):
    """Decorator to enforce role-based access control."""
    async def role_checker(current_user: models.User = Depends(auth.get_current_user)):
        if current_user.role not in allowed_roles and current_user.role != "superadmin":
            # If user is superadmin, they pass everything. Otherwise check list.
            if current_user.role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={"error": "insufficient_permissions"}
                )
        return current_user
    return role_checker

# ─── Tenant Management (Superadmin Only) ───────────────

@admin_router.get("/tenants", dependencies=[Depends(require_role("superadmin"))])
async def list_tenants(db: Session = Depends(get_db)):
    tenants = db.query(
        models.Tenant,
        func.count(models.User.id).label("user_count"),
        func.count(models.Document.id).label("doc_count")
    ).outerjoin(models.User, models.User.tenant_id == models.Tenant.id)\
     .outerjoin(models.Document, models.Document.tenant_id == models.Tenant.id)\
     .group_by(models.Tenant.id).all()
     
    return [
        {
            "id": t.Tenant.id,
            "name": t.Tenant.name,
            "user_count": t.user_count,
            "doc_count": t.doc_count,
            "created_at": str(t.Tenant.created_at)
        } for t in tenants
    ]

@admin_router.post("/tenants", dependencies=[Depends(require_role("superadmin"))])
async def create_tenant(tenant_id: str, tenant_name: str, db: Session = Depends(get_db)):
    # Check if exists
    existing = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Tenant ID already exists")
        
    new_tenant = models.Tenant(id=tenant_id, name=tenant_name)
    db.add(new_tenant)
    db.commit()
    db.refresh(new_tenant)
    return new_tenant

@admin_router.delete("/tenants/{tenant_id}", dependencies=[Depends(require_role("superadmin"))])
async def delete_tenant(tenant_id: str, confirm: str = Query(...), db: Session = Depends(get_db)):
    if confirm != "DELETE":
        raise HTTPException(status_code=400, detail="Must send confirm='DELETE' to confirm deletion")
        
    tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
        
    # 1. Counts for report
    user_count = db.query(models.User).filter(models.User.tenant_id == tenant_id).count()
    doc_count = db.query(models.Document).filter(models.Document.tenant_id == tenant_id).count()
    
    # 2. Delete ChromaDB collection
    import ingestion
    try:
        vector_store = ingestion.get_vector_store(tenant_id)
        vector_store.delete_collection()
        # Remove persist dir
        persist_path = os.path.join(config.CHROMA_DB_PATH, tenant_id)
        if os.path.exists(persist_path):
            shutil.rmtree(persist_path)
    except Exception as e:
        logger.error(f"Error deleting Chroma collection for {tenant_id}: {e}")

    # 3. Delete files from uploads
    for filename in os.listdir(config.UPLOAD_DIR):
        if filename.startswith(f"{tenant_id}_"):
            os.remove(os.path.join(config.UPLOAD_DIR, filename))
            
    # 4. Delete BM25 index
    bm25_path = f"./bm25_indexes/{tenant_id}_bm25.pkl"
    if os.path.exists(bm25_path):
        os.remove(bm25_path)

    # 5. Delete from DB (cascading handles users, documents via SQLAlchemy relationships)
    db.delete(tenant)
    db.commit()
    
    return {
        "deleted_tenant": tenant_id,
        "deleted_users": user_count,
        "deleted_docs": doc_count
    }

# ─── User Management ────────────────────────────────────

@admin_router.get("/users")
async def list_users(
    current_user: models.User = Depends(require_role("admin", "superadmin")),
    db: Session = Depends(get_db)
):
    query = db.query(models.User)
    if current_user.role != "superadmin":
        query = query.filter(models.User.tenant_id == current_user.tenant_id)
        
    users = query.all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "role": u.role,
            "tenant_id": u.tenant_id,
            "created_at": str(datetime.utcnow()), # Placeholder if not in model
            "is_locked": u.locked_until > datetime.utcnow() if u.locked_until else False
        } for u in users
    ]

@admin_router.post("/users")
async def create_user(
    email: str, password: str, role: str, tenant_id: str = None,
    current_user: models.User = Depends(require_role("admin", "superadmin")),
    db: Session = Depends(get_db)
):
    # Admin can only create in their own tenant
    if current_user.role != "superadmin":
        tenant_id = current_user.tenant_id
    elif tenant_id is None:
        raise HTTPException(status_code=400, detail="Superadmin must specify tenant_id")
        
    # Check if email exists
    existing = db.query(models.User).filter(models.User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User with this email already exists")
        
    if role == "superadmin" and current_user.role != "superadmin":
        raise HTTPException(status_code=403, detail="Only superadmins can create other superadmins")
        
    new_user = models.User(
        email=email,
        hashed_password=auth.get_password_hash(password),
        tenant_id=tenant_id,
        role=role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@admin_router.patch("/users/{user_id}/role")
async def update_user_role(
    user_id: int, role: str,
    current_user: models.User = Depends(require_role("admin", "superadmin")),
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if current_user.role != "superadmin":
        if user.tenant_id != current_user.tenant_id:
            raise HTTPException(status_code=403, detail="Cannot edit users from other tenants")
        if role == "superadmin":
            raise HTTPException(status_code=403, detail="Admins cannot promote to superadmin")
            
    user.role = role
    db.commit()
    return user

@admin_router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: models.User = Depends(require_role("admin", "superadmin")),
    db: Session = Depends(get_db)
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
        
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if current_user.role != "superadmin" and user.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=403, detail="Cannot delete users from other tenants")
        
    if user.role == "superadmin":
        # Check if they are the only superadmin? Maybe just block.
        raise HTTPException(status_code=403, detail="Superadmins cannot be deleted")
        
    db.delete(user)
    db.commit()
    return {"message": "User deleted"}

# ─── Audit Log Viewer ───────────────────────────────────

@admin_router.get("/audit-logs")
async def get_audit_logs(
    page: int = 1,
    limit: int = 50,
    event_type: Optional[str] = None,
    tenant_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    current_user: models.User = Depends(require_role("admin", "superadmin"))
):
    limit = min(limit, 200)
    log_file = os.path.join(config.LOG_DIR, "audit.log")
    
    if not os.path.exists(log_file):
        return {"logs": [], "total": 0, "page": page}
        
    # Scoping
    if current_user.role != "superadmin":
        tenant_id = current_user.tenant_id
        
    logs = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Filter
                    if event_type and entry.get("event") != event_type:
                        continue
                    if tenant_id and entry.get("tenant_id") != tenant_id:
                        continue
                    if date_from and entry.get("timestamp") < date_from:
                        continue
                    if date_to and entry.get("timestamp") > date_to:
                        continue
                    logs.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Error reading audit logs: {e}")
        
    # Sort reverse chronological
    logs.sort(key=lambda x: x.get("timestamp"), reverse=True)
    
    total = len(logs)
    start = (page - 1) * limit
    end = start + limit
    
    return {
        "logs": logs[start:end],
        "total": total,
        "page": page
    }

# ─── System Stats ───────────────────────────────────────

@admin_router.get("/stats")
async def get_system_stats(
    current_user: models.User = Depends(require_role("admin", "superadmin")),
    db: Session = Depends(get_db)
):
    # Total tenants
    if current_user.role == "superadmin":
        total_tenants = db.query(models.Tenant).count()
        total_users = db.query(models.User).count()
        total_docs = db.query(models.Document).count()
        total_msgs = db.query(models.ChatMessage).count()
    else:
        total_tenants = 1
        total_users = db.query(models.User).filter(models.User.tenant_id == current_user.tenant_id).count()
        total_docs = db.query(models.Document).filter(models.Document.tenant_id == current_user.tenant_id).count()
        total_msgs = db.query(models.ChatMessage).filter(models.ChatMessage.tenant_id == current_user.tenant_id).count()

    # Storage used (MB)
    storage_bytes = 0
    if current_user.role == "superadmin":
        for f in os.listdir(config.UPLOAD_DIR):
            storage_bytes += os.path.getsize(os.path.join(config.UPLOAD_DIR, f))
    else:
        for f in os.listdir(config.UPLOAD_DIR):
            if f.startswith(f"{current_user.tenant_id}_"):
                storage_bytes += os.path.getsize(os.path.join(config.UPLOAD_DIR, f))
                
    # Vector count
    vector_count = 0
    try:
        vs = ingestion.get_vector_store(current_user.tenant_id)
        vector_count = vs._collection.count()
    except:
        pass

    return {
        "total_tenants": total_tenants,
        "total_users": total_users,
        "total_documents": total_docs,
        "total_chat_messages": total_msgs,
        "storage_used_mb": round(storage_bytes / (1024 * 1024), 2),
        "chroma_collection_size": vector_count,
        "top_queried_docs": [] # Complex to implement without citation parsing logic
    }

# ─── Ollama Model Management ────────────────────────────

@admin_router.get("/models", dependencies=[Depends(require_role("superadmin"))])
async def list_ollama_models():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

@admin_router.post("/models/switch", dependencies=[Depends(require_role("superadmin"))])
async def switch_model(model_name: str):
    # Check if model exists
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            available = [m["name"] for m in response.json().get("models", [])]
            if model_name not in available:
                # Try pull or just fail? Pull is slow. Let's just require it exists.
                raise HTTPException(status_code=400, detail="Model not found in Ollama. Pull it first.")
        except Exception as e:
            if isinstance(e, HTTPException): raise
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")
            
    # Update in-memory config
    config.OLLAMA_LLM_MODEL = model_name
    return {
        "status": "success",
        "active_model": model_name,
        "note": "This change is in-memory only and will reset on server restart."
    }
