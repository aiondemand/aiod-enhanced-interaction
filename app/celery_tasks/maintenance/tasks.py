"""Celery tasks for maintenance operations."""

from __future__ import annotations

import asyncio
import gc
import logging
import torch

from app.celery_app import celery_app
from app.celery_tasks.maintenance.helpers import (
    acquire_lock,
    ensure_worker_db_initialized,
    release_lock,
)
from app.config import settings
from app.services.chatbot.website_scraper import scraping_wrapper
from app.services.threads.db_gc_thread import mongo_cleanup
from app.services.threads.embedding_thread import compute_embeddings_for_aiod_assets_wrapper
from app.services.threads.metadata_extraction_thread import extract_metadata_for_assets_wrapper
from app.services.threads.milvus_gc_thread import delete_embeddings_of_aiod_assets_wrapper
from app.services.resilience import LocalServiceUnavailableException, MilvusUnavailableException


@celery_app.task(
    bind=True,
    max_retries=3,
)
def compute_embeddings_task(self, first_invocation: bool = False) -> dict:
    """
    Task to compute embeddings for AIoD assets.

    Args:
        first_invocation: Whether this is the first invocation (initial setup)

    Returns:
        dict with task result information
    """
    ensure_worker_db_initialized()

    lock_key = "compute_embeddings_lock"

    if not acquire_lock(lock_key, timeout=7200):  # 2 hour timeout
        logging.info(
            "[RECURRING AIOD UPDATE] Scheduled task for computing asset embeddings skipped (previous task is still running)"
        )
        return {"status": "skipped", "reason": "Task already running"}

    try:
        log_msg = (
            "[RECURRING AIOD UPDATE] Initial task for computing asset embeddings has started"
            if first_invocation
            else "[RECURRING AIOD UPDATE] Scheduled task for computing asset embeddings has started"
        )
        logging.info(log_msg)

        # Run async function
        asyncio.run(compute_embeddings_for_aiod_assets_wrapper(first_invocation))

        logging.info(
            "[RECURRING AIOD UPDATE] Scheduled task for computing asset embeddings has ended."
        )

        return {"status": "completed", "first_invocation": first_invocation}
    except LocalServiceUnavailableException as e:
        logging.error(e)
        logging.error(
            "[RECURRING AIOD UPDATE] The above error has been encountered in the embedding task. "
            + "Task will be retried."
        )
        # Re-raise to trigger retry
        raise
    except Exception as e:
        logging.error(e)
        logging.error(
            "[RECURRING AIOD UPDATE] The above error has been encountered in the embedding task."
        )
        return {"status": "failed", "error": str(e)}
    finally:
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        release_lock(lock_key)


@celery_app.task(
    bind=True,
    max_retries=3,
)
def delete_embeddings_task(self) -> dict:
    """
    Task to delete embeddings of removed AIoD assets (garbage collection).

    Returns:
        dict with task result information
    """
    ensure_worker_db_initialized()

    lock_key = "delete_embeddings_lock"

    if not acquire_lock(lock_key, timeout=7200):  # 2 hour timeout
        logging.info(
            "[RECURRING MILVUS DELETE] Scheduled task for deleting skipped (previous task is still running)"
        )
        return {"status": "skipped", "reason": "Task already running"}

    try:
        logging.info(
            "[RECURRING MILVUS DELETE] Scheduled task for deleting asset embeddings has started."
        )

        # Run async function
        asyncio.run(delete_embeddings_of_aiod_assets_wrapper())

        logging.info(
            "[RECURRING MILVUS DELETE] Scheduled task for deleting asset embeddings has ended."
        )

        return {"status": "completed"}
    except MilvusUnavailableException as e:
        logging.error(e)
        logging.error(
            "[RECURRING MILVUS DELETE] The above error has been encountered in the Milvus garbage collection task. "
            + "Task will be retried."
        )
        # Re-raise to trigger retry
        raise
    except Exception as e:
        logging.error(e)
        logging.error(
            "[RECURRING MILVUS DELETE] The above error has been encountered in the Milvus garbage collection task."
        )
        return {"status": "failed", "error": str(e)}
    finally:
        release_lock(lock_key)


@celery_app.task(
    bind=True,
    max_retries=3,
)
def mongo_cleanup_task(self) -> dict:
    """
    Task to clean up expired queries and empty asset collections from MongoDB.

    Returns:
        dict with task result information
    """
    ensure_worker_db_initialized()

    lock_key = "mongo_cleanup_lock"

    if not acquire_lock(lock_key, timeout=3600):  # 1 hour timeout
        logging.info(
            "[RECURRING MONGODB DELETE] Scheduled task for cleaning up MongoDB skipped (previous task is still running)"
        )
        return {"status": "skipped", "reason": "Task already running"}

    try:
        logging.info(
            "[RECURRING MONGODB DELETE] Scheduled task for cleaning up MongoDB has started."
        )

        # Run async function
        asyncio.run(mongo_cleanup())

        logging.info("[RECURRING MONGODB DELETE] Scheduled task for cleaning up MongoDB has ended.")

        return {"status": "completed"}
    except Exception as e:
        logging.error(e)
        logging.error(
            "[RECURRING MONGODB DELETE] The above error has been encountered in the MongoDB cleanup task."
        )
        return {"status": "failed", "error": str(e)}
    finally:
        release_lock(lock_key)


@celery_app.task(
    bind=True,
    max_retries=3,
)
def extract_metadata_task(self) -> dict:
    """
    Task to extract metadata from AIoD assets using LLM.

    Returns:
        dict with task result information
    """
    if not settings.PERFORM_METADATA_EXTRACTION:
        return {"status": "skipped", "reason": "Metadata extraction disabled"}

    ensure_worker_db_initialized()

    lock_key = "extract_metadata_lock"

    if not acquire_lock(lock_key, timeout=7200):  # 2 hour timeout
        logging.info(
            "[RECURRING METADATA EXTRACTION] Scheduled task for metadata extraction skipped (previous task is still running)"
        )
        return {"status": "skipped", "reason": "Task already running"}

    try:
        logging.info(
            "[RECURRING METADATA EXTRACTION] Scheduled task for extracting asset metadata has started"
        )

        # Run async function
        asyncio.run(extract_metadata_for_assets_wrapper())

        logging.info(
            "[RECURRING METADATA EXTRACTION] Scheduled task for extracting asset metadata has ended."
        )

        return {"status": "completed"}
    except LocalServiceUnavailableException as e:
        logging.error(e)
        logging.error(
            "[RECURRING METADATA EXTRACTION] The above error has been encountered in the metadata extraction task. "
            + "Task will be retried."
        )
        raise
    except Exception as e:
        logging.error(e)
        logging.error(
            "[RECURRING METADATA EXTRACTION] The above error has been encountered in the metadata extraction task."
        )
        return {"status": "failed", "error": str(e)}
    finally:
        release_lock(lock_key)


@celery_app.task(
    bind=True,
    max_retries=3,
)
def scraping_task(self) -> dict:
    """
    Task to scrape AIoD websites and APIs.

    Returns:
        dict with task result information
    """
    if not settings.CHATBOT.USE_CHATBOT:
        return {"status": "skipped", "reason": "Chatbot disabled"}

    ensure_worker_db_initialized()

    lock_key = "scraping_lock"

    if not acquire_lock(lock_key, timeout=7200):  # 2 hour timeout
        logging.info(
            "Scheduled task for scraping AIoD websites and APIs skipped (previous task is still running)"
        )
        return {"status": "skipped", "reason": "Task already running"}

    try:
        logging.info(
            "[RECURRING SCRAPING] Scheduled task for scraping AIoD websites and APIs has started."
        )

        # Run async function
        asyncio.run(scraping_wrapper())

        logging.info(
            "[RECURRING SCRAPING] Scheduled task for scraping AIoD websites and APIs has ended."
        )

        return {"status": "completed"}
    except Exception as e:
        logging.error(e)
        logging.error(
            "[RECURRING SCRAPING] The above error has been encountered in the scraping task."
        )
        return {"status": "failed", "error": str(e)}
    finally:
        release_lock(lock_key)
