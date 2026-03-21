import os
import hashlib
import logging
import threading
import time
import shutil
import hmac
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import schedule
import asyncio

from database import SessionLocal
import models
import ingestion
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Checksum Detection ──────────────────────────────────

def compute_checksum(file_path: str) -> str:
    return ingestion.compute_checksum(file_path)

def has_file_changed(filename: str, tenant_id: str, new_checksum: str, db: Session) -> bool:
    """Check if file has changed or is new for a tenant."""
    doc = db.query(models.Document).filter(
        models.Document.filename == filename,
        models.Document.tenant_id == tenant_id
    ).first()
    
    if not doc:
        return True # New file
    
    return doc.checksum != new_checksum

# ─── Versioned Re-ingestion ─────────────────────────────

async def reingest_document(doc_id: int, tenant_id: str, file_path: str, db: Session) -> Dict[str, Any]:
    """Handle versioned re-ingestion of a document."""
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        return {"status": "error", "message": "Document not found"}
        
    # Check file size (max 100MB)
    if os.path.getsize(file_path) > 100 * 1024 * 1024:
        return {"status": "error", "message": "File too large (max 100MB)"}
        
    new_checksum = compute_checksum(file_path)
    if doc.checksum == new_checksum:
        return {"status": "unchanged", "message": "File content is identical"}
        
    # 1. Soft-delete old versions in ChromaDB
    # Note: We need to use the vector_store directly to update metadata
    from ingestion import get_vector_store
    vector_store = get_vector_store(tenant_id)
    
    # Chroma doesn't have a direct 'update by filter' for metadata in all versions,
    # so we might need to get IDs and update them.
    # For simplicity, we assume we can update by doc_id filter if supported, 
    # otherwise we get and update.
    try:
        results = vector_store.get(where={"doc_id": doc_id, "active": True})
        if results["ids"]:
            # Update each chunk to active=False
            for i, chunk_id in enumerate(results["ids"]):
                meta = results["metadatas"][i]
                meta["active"] = False
                vector_store.update_metadata(chunk_id, meta)
    except Exception as e:
        logger.error(f"Error soft-deleting old chunks for doc {doc_id}: {e}")

    # 2. Increment version and re-ingest
    old_version = doc.version
    doc.version += 1
    doc.checksum = new_checksum
    doc.updated_at = datetime.utcnow()
    
    # Re-run ingestion
    chunk_count = await ingestion.process_document(file_path, tenant_id, doc_id, version=doc.version)
    
    # 3. Record version history
    version_record = models.DocumentVersion(
        doc_id=doc_id,
        version_number=doc.version,
        checksum=new_checksum,
        chunk_count=chunk_count,
        notes=f"Updated from version {old_version}"
    )
    db.add(version_record)
    db.commit()
    
    return {"status": "updated", "new_version": doc.version, "chunks": chunk_count}

# ─── Rollback ───────────────────────────────────────────

def rollback_document(doc_id: int, target_version: int, tenant_id: str, db: Session) -> Dict[str, Any]:
    """Rollback document to a previous version."""
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if not doc:
        return {"status": "error", "message": "Document not found"}
        
    version_exists = db.query(models.DocumentVersion).filter(
        models.DocumentVersion.doc_id == doc_id,
        models.DocumentVersion.version_number == target_version
    ).first()
    
    if not version_exists:
        return {"status": "error", "message": f"Version {target_version} not found"}
        
    # Toggle active status in ChromaDB
    from ingestion import get_vector_store
    vector_store = get_vector_store(tenant_id)
    
    try:
        # Deactivate current
        current_results = vector_store.get(where={"doc_id": doc_id, "active": True})
        if current_results["ids"]:
            for i, chunk_id in enumerate(current_results["ids"]):
                meta = current_results["metadatas"][i]
                meta["active"] = False
                vector_store.update_metadata(chunk_id, meta)
                
        # Activate target
        target_results = vector_store.get(where={"doc_id": doc_id, "version": target_version})
        if target_results["ids"]:
            for i, chunk_id in enumerate(target_results["ids"]):
                meta = target_results["metadatas"][i]
                meta["active"] = True
                vector_store.update_metadata(chunk_id, meta)
                
        doc.version = target_version
        db.commit()
        return {"status": "rolled_back", "version": target_version}
    except Exception as e:
        logger.error(f"Rollback error for doc {doc_id}: {e}")
        return {"status": "error", "message": str(e)}

# ─── Folder Watcher ──────────────────────────────────────

class IngestionHandler(FileSystemEventHandler):
    def __init__(self, thread_pool_executor=None):
        self.executor = thread_pool_executor

    def on_modified(self, event):
        if not event.is_directory:
            self._handle_event(event.src_path, "update")

    def on_created(self, event):
        if not event.is_directory:
            self._handle_event(event.src_path, "add")

    def on_deleted(self, event):
        if not event.is_directory:
            self._handle_event(event.src_path, "delete")

    def _handle_event(self, path, action):
        ext = os.path.splitext(path)[1].lower()
        if ext not in [".pdf", ".docx", ".txt", ".csv"]:
            return

        # Path structure: watched_folders/{tenant_id}/filename
        parts = os.path.normpath(path).split(os.sep)
        if len(parts) < 3:
            return
            
        tenant_id = parts[-2]
        filename = parts[-1]
        
        db = SessionLocal()
        try:
            tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
            if not tenant:
                logger.warning(f"Tenant {tenant_id} not found for file {filename}")
                return

            if action == "delete":
                doc = db.query(models.Document).filter(
                    models.Document.filename == filename,
                    models.Document.tenant_id == tenant_id
                ).first()
                if doc:
                    # Soft delete vectors and DB record
                    from ingestion import get_vector_store
                    vector_store = get_vector_store(tenant_id)
                    vector_store.delete(where={"doc_id": doc.id})
                    db.delete(doc)
                    db.commit()
                    logger.info(f"Deleted document: {filename} for tenant {tenant_id}")
            else:
                checksum = compute_checksum(path)
                if has_file_changed(filename, tenant_id, checksum, db):
                    logger.info(f"Auto-ingesting: {filename} for tenant {tenant_id}")
                    # In a real app, we'd use a background task runner.
                    # Here we'll just run it in a thread for the watcher.
                    doc = db.query(models.Document).filter(
                        models.Document.filename == filename,
                        models.Document.tenant_id == tenant_id
                    ).first()
                    
                    if not doc:
                        # Initial ingestion - need to copy to uploads first
                        upload_path = os.path.join(config.UPLOAD_DIR, f"{tenant_id}_{filename}")
                        shutil.copy2(path, upload_path)
                        # We'd normally create a record here, but we'll assume the 
                        # upload route's logic or a simplified version.
                        # For the watcher, we'll create the record if missing.
                        doc = models.Document(filename=filename, file_type=ext[1:], tenant_id=tenant_id, checksum=checksum)
                        db.add(doc)
                        db.commit()
                        db.refresh(doc)
                        asyncio.run(ingestion.process_document(upload_path, tenant_id, doc.id))
                    else:
                        upload_path = os.path.join(config.UPLOAD_DIR, f"{tenant_id}_{filename}")
                        shutil.copy2(path, upload_path)
                        asyncio.run(reingest_document(doc.id, tenant_id, upload_path, db))
        except Exception as e:
            logger.error(f"Watcher error for {path}: {e}")
        finally:
            db.close()

def start_folder_watcher():
    """Start the watchdog observer in a daemon thread."""
    try:
        os.makedirs(config.WATCHED_FOLDERS, exist_ok=True)
        event_handler = IngestionHandler()
        observer = Observer()
        observer.schedule(event_handler, config.WATCHED_FOLDERS, recursive=True)
        observer.start()
        logger.info(f"Folder watcher started on {config.WATCHED_FOLDERS}")
        
        # Keep the thread alive if needed, but Observer runs its own thread.
        # We just need to make sure the main process doesn't exit if it's the only thread.
    except ImportError:
        logger.warning("watchdog not installed. Skipping folder watcher.")
    except Exception as e:
        logger.error(f"Failed to start folder watcher: {e}")

# ─── Scheduled Sync ────────────────────────────────────

def run_scheduled_sync():
    """Scan watched folders and update changed files."""
    db = SessionLocal()
    try:
        updated_count = 0
        total_checked = 0
        
        if not os.path.exists(config.WATCHED_FOLDERS):
            return

        for tenant_id in os.listdir(config.WATCHED_FOLDERS):
            tenant_path = os.path.join(config.WATCHED_FOLDERS, tenant_id)
            if not os.path.isdir(tenant_path):
                continue
                
            tenant = db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()
            if not tenant:
                continue

            for filename in os.listdir(tenant_path):
                file_path = os.path.join(tenant_path, filename)
                if os.path.isdir(file_path):
                    continue
                    
                ext = os.path.splitext(filename)[1].lower()
                if ext not in [".pdf", ".docx", ".txt", ".csv"]:
                    continue

                total_checked += 1
                checksum = compute_checksum(file_path)
                if has_file_changed(filename, tenant_id, checksum, db):
                    upload_path = os.path.join(config.UPLOAD_DIR, f"{tenant_id}_{filename}")
                    shutil.copy2(file_path, upload_path)
                    
                    doc = db.query(models.Document).filter(
                        models.Document.filename == filename,
                        models.Document.tenant_id == tenant_id
                    ).first()
                    
                    if not doc:
                        doc = models.Document(filename=filename, file_type=ext[1:], tenant_id=tenant_id, checksum=checksum)
                        db.add(doc)
                        db.commit()
                        db.refresh(doc)
                        asyncio.run(ingestion.process_document(upload_path, tenant_id, doc.id))
                    else:
                        asyncio.run(reingest_document(doc.id, tenant_id, upload_path, db))
                    updated_count += 1

        logger.info(f"Sync complete: {total_checked} files checked, {updated_count} updated")
    except Exception as e:
        logger.error(f"Scheduled sync error: {e}")
    finally:
        db.close()

def start_scheduler():
    """Start the schedule runner in a daemon thread."""
    try:
        schedule.every(config.SYNC_INTERVAL_MINUTES).minutes.do(run_scheduled_sync)
        logger.info(f"Scheduled sync started every {config.SYNC_INTERVAL_MINUTES} minutes")
        
        def run_loop():
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
    except ImportError:
        logger.warning("schedule not installed. Skipping scheduled sync.")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
