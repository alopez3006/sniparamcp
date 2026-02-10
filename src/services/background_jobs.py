"""Background job processor for async document indexing.

This service handles:
1. Polling for pending index jobs
2. Claiming and processing jobs atomically
3. Updating progress during indexing
4. Handling failures and retries
5. Detecting stale jobs from crashed workers
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prisma.models import IndexJob

    from prisma import Prisma

logger = logging.getLogger(__name__)

# Configuration
JOB_POLL_INTERVAL = 10  # seconds between polling
JOB_STALE_TIMEOUT = 300  # 5 min - reclaim if worker crashed
MAX_CONCURRENT_JOBS = 2  # per worker
AUTO_DISCOVERY_INTERVAL = 300  # 5 min - scan for unindexed documents

# Global state
_running = False
_processor_task: asyncio.Task | None = None
_discovery_task: asyncio.Task | None = None
_active_jobs: set[str] = set()
_worker_id: str = f"worker-{os.getpid()}-{uuid.uuid4().hex[:8]}"
_last_discovery: datetime | None = None


async def start_job_processor(db: Prisma) -> None:
    """Start the background job processor loop."""
    global _running, _processor_task, _discovery_task

    if _running:
        logger.warning("Job processor already running")
        return

    _running = True
    _processor_task = asyncio.create_task(_job_processor_loop(db))
    _discovery_task = asyncio.create_task(_auto_discovery_loop(db))
    logger.info(f"Job processor started (worker_id={_worker_id})")


async def stop_job_processor() -> None:
    """Stop the background job processor."""
    global _running, _processor_task, _discovery_task

    _running = False
    for task in [_processor_task, _discovery_task]:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    _processor_task = None
    _discovery_task = None
    logger.info("Job processor stopped")


async def _job_processor_loop(db: Prisma) -> None:
    """Main loop that polls for and processes jobs."""
    while _running:
        try:
            # Only claim new jobs if we have capacity
            if len(_active_jobs) < MAX_CONCURRENT_JOBS:
                job = await _claim_next_job(db)
                if job:
                    # Process in background, don't block the loop
                    asyncio.create_task(_process_job_wrapper(db, job))
        except Exception as e:
            logger.error(f"Error in job processor loop: {e}")

        await asyncio.sleep(JOB_POLL_INTERVAL)


async def _auto_discovery_loop(db: Prisma) -> None:
    """
    Periodically scan for projects with unindexed documents and create index jobs.

    This ensures all projects eventually get indexed, even if the webhook-based
    indexing fails (e.g., due to misconfigured MCP_INTERNAL_SECRET).
    """
    global _last_discovery

    # Wait a bit before first discovery to let the server stabilize
    await asyncio.sleep(60)

    while _running:
        try:
            now = datetime.now(UTC)

            # Find projects with documents that have no chunks
            # Use a single efficient query that:
            # 1. Groups by project
            # 2. Counts docs with 0 chunks
            # 3. Only returns projects with unindexed docs
            # 4. Excludes projects that already have pending/running jobs
            projects_needing_index = await db.query_raw(
                """
                WITH doc_stats AS (
                    SELECT
                        d."projectId",
                        COUNT(*) FILTER (
                            WHERE NOT EXISTS (
                                SELECT 1 FROM document_chunks dc WHERE dc."documentId" = d.id
                            )
                        ) as unindexed_count
                    FROM documents d
                    GROUP BY d."projectId"
                    HAVING COUNT(*) FILTER (
                        WHERE NOT EXISTS (
                            SELECT 1 FROM document_chunks dc WHERE dc."documentId" = d.id
                        )
                    ) > 0
                ),
                pending_jobs AS (
                    SELECT DISTINCT "projectId"
                    FROM index_jobs
                    WHERE status IN ('PENDING', 'RUNNING')
                )
                SELECT
                    ds."projectId",
                    ds.unindexed_count,
                    p.name as project_name,
                    p.slug as project_slug
                FROM doc_stats ds
                JOIN projects p ON ds."projectId" = p.id
                LEFT JOIN pending_jobs pj ON ds."projectId" = pj."projectId"
                WHERE pj."projectId" IS NULL
                ORDER BY ds.unindexed_count DESC
                LIMIT 10
                """
            )

            if projects_needing_index:
                logger.info(
                    f"Auto-discovery found {len(projects_needing_index)} projects "
                    f"with unindexed documents"
                )

                for proj in projects_needing_index:
                    try:
                        # Create an incremental index job
                        result = await db.query_raw(
                            """
                            INSERT INTO index_jobs
                            (id, "projectId", status, progress, "indexMode", "createdAt", "updatedAt", "triggeredBy", "triggeredVia")
                            VALUES (gen_random_uuid()::text, $1, 'PENDING', 0, 'INCREMENTAL'::"IndexJobMode", NOW(), NOW(), 'system', 'auto_discovery')
                            RETURNING id
                            """,
                            proj["projectId"],
                        )
                        job_id = result[0]["id"] if result else None
                        logger.info(
                            f"Created auto-discovery index job {job_id} for "
                            f"project {proj['project_slug']} ({proj['unindexed_count']} unindexed docs)"
                        )
                    except Exception as e:
                        # Job might already exist due to race condition - that's OK
                        if "unique constraint" not in str(e).lower():
                            logger.error(
                                f"Failed to create index job for {proj['project_slug']}: {e}"
                            )

            _last_discovery = now

        except Exception as e:
            logger.error(f"Error in auto-discovery loop: {e}")

        await asyncio.sleep(AUTO_DISCOVERY_INTERVAL)


async def _claim_next_job(db: Prisma) -> IndexJob | None:
    """
    Atomically claim the next available job.

    Claims either:
    1. PENDING jobs (never started)
    2. Stale RUNNING jobs (worker crashed)
    """
    now = datetime.now(UTC)
    stale_cutoff = now - timedelta(seconds=JOB_STALE_TIMEOUT)

    # Try to claim a pending job or a stale running job
    # Using raw SQL for atomic update with RETURNING
    # Note: Convert datetime to ISO string for Prisma query_raw compatibility
    result = await db.query_raw(
        """
        UPDATE index_jobs
        SET status = 'RUNNING',
            "workerId" = $1,
            "startedAt" = CASE WHEN "startedAt" IS NULL THEN $2::timestamp ELSE "startedAt" END,
            "updatedAt" = $2::timestamp
        WHERE id = (
            SELECT id FROM index_jobs
            WHERE (status = 'PENDING')
               OR (status = 'RUNNING' AND "updatedAt" < $3::timestamp AND "retryCount" < "maxRetries")
            ORDER BY
                CASE WHEN status = 'PENDING' THEN 0 ELSE 1 END,
                "createdAt" ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, "projectId", status, progress, "indexMode", "documentsTotal", "documentsProcessed",
                  "chunksCreated", "retryCount", "maxRetries", "errorMessage",
                  "createdAt", "startedAt", "completedAt", "updatedAt",
                  "triggeredBy", "triggeredVia", "workerId", results
        """,
        _worker_id,
        now.isoformat(),
        stale_cutoff.isoformat(),
    )

    if not result:
        return None

    row = result[0]
    logger.info(f"Claimed job {row['id']} for project {row['projectId']}")

    # If this was a stale job, increment retry count
    if row.get("retryCount", 0) > 0 or row.get("errorMessage"):
        await db.execute_raw(
            """
            UPDATE index_jobs
            SET "retryCount" = "retryCount" + 1, "errorMessage" = NULL
            WHERE id = $1
            """,
            row["id"],
        )

    # Return the job data as a dict (we'll use it directly)
    return row


async def _process_job_wrapper(db: Prisma, job: dict) -> None:
    """Wrapper to track active jobs and handle exceptions."""
    job_id = job["id"]
    _active_jobs.add(job_id)

    try:
        await _process_index_job(db, job)
    except Exception as e:
        logger.error(f"Job {job_id} failed with error: {e}")
        await _fail_job(db, job_id, str(e))
    finally:
        _active_jobs.discard(job_id)


async def _process_index_job(db: Prisma, job: dict) -> None:
    """Process an index job by indexing project documents.

    Supports two modes:
    - FULL: Re-index all documents (deletes existing chunks)
    - INCREMENTAL: Only index documents without existing chunks
    """
    from .indexer import DocumentIndexer

    job_id = job["id"]
    project_id = job["projectId"]
    index_mode = job.get("indexMode", "INCREMENTAL")  # Default to incremental

    logger.info(f"Processing index job {job_id} for project {project_id} (mode={index_mode})")

    # Create indexer
    indexer = DocumentIndexer(db)

    if index_mode == "INCREMENTAL":
        # Get only documents without chunks
        documents = await db.query_raw(
            '''
            SELECT d.id, d.path
            FROM documents d
            LEFT JOIN document_chunks dc ON d.id = dc."documentId"
            WHERE d."projectId" = $1
            GROUP BY d.id, d.path
            HAVING COUNT(dc.id) = 0
            ORDER BY d.path
            ''',
            project_id,
        )
        # Convert to list of dicts with id/path
        documents = [{"id": doc["id"], "path": doc["path"]} for doc in documents]
    else:
        # Get all documents
        docs = await db.document.find_many(where={"projectId": project_id})
        documents = [{"id": doc.id, "path": doc.path} for doc in docs]

    total_docs = len(documents)

    if total_docs == 0:
        logger.info(f"No documents to index for job {job_id} (mode={index_mode})")
        await _complete_job(db, job_id, 0, 0, {})
        return

    # Update total count
    await db.execute_raw(
        """
        UPDATE index_jobs
        SET "documentsTotal" = $1, "updatedAt" = NOW()
        WHERE id = $2
        """,
        total_docs,
        job_id,
    )

    # Index each document with progress updates
    results: dict[str, int] = {}
    total_chunks = 0
    processed = 0

    for doc in documents:
        try:
            chunk_count = await indexer.index_document(doc["id"])
            results[doc["path"]] = chunk_count
            total_chunks += chunk_count
        except Exception as e:
            logger.error(f"Error indexing document {doc['path']}: {e}")
            results[doc["path"]] = 0

        processed += 1

        # Update progress
        progress = int((processed / total_docs) * 100)
        await _update_progress(db, job_id, progress, processed, total_chunks)

    # Complete the job
    await _complete_job(db, job_id, total_docs, total_chunks, results)


async def _update_progress(
    db: Prisma, job_id: str, progress: int, docs_processed: int, chunks_created: int
) -> None:
    """Update job progress."""
    await db.execute_raw(
        """
        UPDATE index_jobs
        SET progress = $1,
            "documentsProcessed" = $2,
            "chunksCreated" = $3,
            "updatedAt" = NOW()
        WHERE id = $4 AND "workerId" = $5
        """,
        progress,
        docs_processed,
        chunks_created,
        job_id,
        _worker_id,
    )


async def _complete_job(
    db: Prisma, job_id: str, docs_indexed: int, chunks_created: int, results: dict
) -> None:
    """Mark a job as completed."""
    import json

    await db.execute_raw(
        """
        UPDATE index_jobs
        SET status = 'COMPLETED',
            progress = 100,
            "documentsProcessed" = $1,
            "chunksCreated" = $2,
            "completedAt" = NOW(),
            "updatedAt" = NOW(),
            results = $3::jsonb
        WHERE id = $4 AND "workerId" = $5
        """,
        docs_indexed,
        chunks_created,
        json.dumps(results),
        job_id,
        _worker_id,
    )
    logger.info(f"Job {job_id} completed: {docs_indexed} docs, {chunks_created} chunks")


async def _fail_job(db: Prisma, job_id: str, error_message: str) -> None:
    """Mark a job as failed."""
    # Check if we should retry
    result = await db.query_raw(
        """
        SELECT "retryCount", "maxRetries"
        FROM index_jobs
        WHERE id = $1
        """,
        job_id,
    )

    if result:
        row = result[0]
        retry_count = row.get("retryCount", 0)
        max_retries = row.get("maxRetries", 3)

        if retry_count < max_retries:
            # Reset to PENDING for retry
            await db.execute_raw(
                """
                UPDATE index_jobs
                SET status = 'PENDING',
                    "errorMessage" = $1,
                    "workerId" = NULL,
                    "updatedAt" = NOW()
                WHERE id = $2
                """,
                error_message,
                job_id,
            )
            logger.info(f"Job {job_id} failed, will retry ({retry_count + 1}/{max_retries})")
        else:
            # Max retries exceeded, mark as failed
            await db.execute_raw(
                """
                UPDATE index_jobs
                SET status = 'FAILED',
                    "errorMessage" = $1,
                    "completedAt" = NOW(),
                    "updatedAt" = NOW()
                WHERE id = $2
                """,
                error_message,
                job_id,
            )
            logger.error(f"Job {job_id} failed permanently after {retry_count} retries")


async def create_index_job(
    db: Prisma,
    project_id: str,
    triggered_by: str | None = None,
    triggered_via: str | None = None,
    index_mode: str = "INCREMENTAL",
) -> dict:
    """
    Create a new index job for a project.

    Args:
        db: Prisma database connection.
        project_id: The project ID to index.
        triggered_by: User ID or "system", "webhook".
        triggered_via: "api_key", "internal", "dashboard".
        index_mode: "INCREMENTAL" (only unindexed docs) or "FULL" (all docs).

    Returns the created job data.
    """
    # Validate index_mode
    if index_mode not in ("INCREMENTAL", "FULL"):
        index_mode = "INCREMENTAL"

    # Check if there's already a pending/running job for this project
    existing = await db.query_raw(
        """
        SELECT id, status, progress, "indexMode", "createdAt"
        FROM index_jobs
        WHERE "projectId" = $1 AND status IN ('PENDING', 'RUNNING')
        ORDER BY "createdAt" DESC
        LIMIT 1
        """,
        project_id,
    )

    if existing:
        row = existing[0]
        logger.info(f"Index job already exists for project {project_id}: {row['id']}")
        return {
            "id": row["id"],
            "project_id": project_id,
            "status": row["status"].lower(),
            "progress": row["progress"],
            "index_mode": row.get("indexMode", "INCREMENTAL"),
            "created_at": row["createdAt"] if row["createdAt"] else None,
            "already_exists": True,
        }

    # Create new job
    result = await db.query_raw(
        """
        INSERT INTO index_jobs (id, "projectId", status, progress, "indexMode", "createdAt", "updatedAt", "triggeredBy", "triggeredVia")
        VALUES (gen_random_uuid()::text, $1, 'PENDING', 0, $2::"IndexJobMode", NOW(), NOW(), $3, $4)
        RETURNING id, "projectId", status, progress, "indexMode", "createdAt"
        """,
        project_id,
        index_mode,
        triggered_by,
        triggered_via,
    )

    row = result[0]
    logger.info(f"Created index job {row['id']} for project {project_id} (mode={index_mode})")

    return {
        "id": row["id"],
        "project_id": row["projectId"],
        "status": row["status"].lower(),
        "progress": row["progress"],
        "index_mode": row["indexMode"],
        "created_at": row["createdAt"] if row["createdAt"] else None,
        "already_exists": False,
    }


async def get_job_status(db: Prisma, project_id: str, job_id: str) -> dict | None:
    """Get the status of an index job."""
    result = await db.query_raw(
        """
        SELECT id, "projectId", status, progress, "indexMode", "errorMessage",
               "documentsTotal", "documentsProcessed", "chunksCreated",
               "retryCount", "maxRetries", "workerId",
               "createdAt", "startedAt", "completedAt", "updatedAt",
               "triggeredBy", "triggeredVia", results
        FROM index_jobs
        WHERE id = $1 AND "projectId" = $2
        """,
        job_id,
        project_id,
    )

    if not result:
        return None

    row = result[0]
    return {
        "id": row["id"],
        "project_id": row["projectId"],
        "status": row["status"].lower(),
        "progress": row["progress"],
        "index_mode": row.get("indexMode", "INCREMENTAL"),
        "error_message": row["errorMessage"],
        "documents_total": row["documentsTotal"],
        "documents_processed": row["documentsProcessed"],
        "chunks_created": row["chunksCreated"],
        "retry_count": row["retryCount"],
        "max_retries": row["maxRetries"],
        "worker_id": row["workerId"],
        "created_at": row["createdAt"] if row["createdAt"] else None,
        "started_at": row["startedAt"] if row["startedAt"] else None,
        "completed_at": row["completedAt"] if row["completedAt"] else None,
        "updated_at": row["updatedAt"] if row["updatedAt"] else None,
        "triggered_by": row["triggeredBy"],
        "triggered_via": row["triggeredVia"],
        "results": row["results"],
    }


async def cancel_job(db: Prisma, project_id: str, job_id: str) -> bool:
    """Cancel a pending or running job."""
    result = await db.execute_raw(
        """
        UPDATE index_jobs
        SET status = 'CANCELLED',
            "completedAt" = NOW(),
            "updatedAt" = NOW()
        WHERE id = $1 AND "projectId" = $2 AND status IN ('PENDING', 'RUNNING')
        """,
        job_id,
        project_id,
    )

    # execute_raw returns number of affected rows
    cancelled = result > 0 if result else False
    if cancelled:
        logger.info(f"Cancelled job {job_id}")
    return cancelled
