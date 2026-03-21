"""
task_queue.py — Async background ingestion queue.
Uses Python's built-in asyncio.Queue (no Redis/Celery needed).
Jobs are persisted in SQLite via the IngestionJob model.
"""
import asyncio
import logging
import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Singleton Queue ────────────────────────────────────────────────────────

_queue: asyncio.Queue = asyncio.Queue()

def get_queue() -> asyncio.Queue:
    return _queue


# ─── Enqueue Helper ─────────────────────────────────────────────────────────

async def enqueue_ingestion(
    job_id: int,
    file_path: str,
    tenant_id: str,
    doc_id: int,
    version_record_id: int,
    version: int = 1
):
    """Add an ingestion task to the queue."""
    await _queue.put({
        "job_id": job_id,
        "file_path": file_path,
        "tenant_id": tenant_id,
        "doc_id": doc_id,
        "version_record_id": version_record_id,
        "version": version,
    })
    logger.info(f"[TaskQueue] Enqueued job {job_id} for doc {doc_id} (tenant: {tenant_id})")


# ─── Background Worker ──────────────────────────────────────────────────────

async def ingestion_worker(db_factory):
    """
    Long-running coroutine started at app startup.
    Pulls tasks from the queue one-by-one and processes them.
    """
    import ingestion as ing
    import models

    logger.info("[TaskQueue] Worker started and listening for jobs…")

    while True:
        task = await _queue.get()
        job_id = task["job_id"]
        file_path = task["file_path"]
        tenant_id = task["tenant_id"]
        doc_id = task["doc_id"]
        version_record_id = task["version_record_id"]
        version = task["version"]

        db = db_factory()
        try:
            # Mark job as processing
            job = db.query(models.IngestionJob).filter(models.IngestionJob.id == job_id).first()
            if job:
                job.status = "processing"
                job.progress_pct = 10
                db.commit()

            logger.info(f"[TaskQueue] Processing job {job_id}: {file_path}")

            # Run actual ingestion (embedding + chunking — this is the slow part)
            chunk_count = await ing.process_document(file_path, tenant_id, doc_id, version=version)

            # Update version record chunk count
            version_rec = db.query(models.DocumentVersion).filter(
                models.DocumentVersion.id == version_record_id
            ).first()
            if version_rec:
                version_rec.chunk_count = chunk_count
                db.commit()

            # Mark job as done
            if job:
                job.status = "done"
                job.progress_pct = 100
                job.completed_at = datetime.datetime.utcnow()
                db.commit()

            logger.info(f"[TaskQueue] Job {job_id} completed — {chunk_count} chunks ingested.")

        except Exception as e:
            logger.error(f"[TaskQueue] Job {job_id} failed: {e}")
            try:
                if job:
                    job.status = "failed"
                    job.error_message = str(e)[:500]
                    job.completed_at = datetime.datetime.utcnow()
                    db.commit()
            except Exception:
                pass

        finally:
            db.close()
            _queue.task_done()
