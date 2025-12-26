import asyncio
import gc
import logging

from celery.signals import worker_process_init
import torch

from app.celery_app import celery_app
from app.celery_tasks.maintenance.task_decorators import maintenance_task
from app.config import settings
from app.services.database import init_mongo_client
from app.services.embedding_store import MilvusEmbeddingStore
from app.services.helper import utc_now
from app.services.inference.model import AiModel
from app.services.resilience import LocalServiceUnavailableException
from app.celery_tasks.maintenance.jobs.clean_mongo_job import clean_mongo_database
from app.celery_tasks.maintenance.jobs.clean_miluvs_job import delete_asset_embeddings
from app.celery_tasks.maintenance.jobs.metadata_extraction_job import extract_metadata_for_assets
from app.celery_tasks.maintenance.jobs.website_scraper_job import populate_collections_wrapper
from app.celery_tasks.maintenance.jobs.new_embeddings_job import compute_embeddings_for_aiod_assets


@worker_process_init.connect
def initialize_each_maintenance_worker_child_process(*args, **kwargs) -> None:
    asyncio.run(init_mongo_client())


def _gpu_cleanup(context: dict) -> None:
    """Cleanup GPU memory after embedding computation."""
    if "model" in context:
        context["model"].to_device("cpu")
        del context["model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _embedding_start_message(kwargs: dict) -> str:
    """Generate custom start message for embedding task."""
    first_invocation = kwargs.get("first_invocation", False)
    if first_invocation:
        return "Initial task for computing asset embeddings has started"
    return "Scheduled task for computing asset embeddings has started"


@celery_app.task(bind=True, acks_late=False, task_reject_on_worker_lost=False)
@maintenance_task(
    log_prefix="[RECURRING AIOD UPDATE]",
    task_description="embedding task",
    non_retryable_exceptions=(LocalServiceUnavailableException,),
    cleanup_func=_gpu_cleanup,
    start_message_func=_embedding_start_message,
)
def compute_embeddings_task(self, first_invocation: bool = False, **kwargs) -> dict:
    context = kwargs["context"]
    model = AiModel(device=AiModel.get_device())
    context["model"] = model  # Store for cleanup

    asyncio.run(compute_embeddings_for_aiod_assets(model, first_invocation))

    return {"first_invocation": first_invocation}


@celery_app.task(bind=True, acks_late=False, task_reject_on_worker_lost=False)
@maintenance_task(
    log_prefix="[RECURRING MILVUS DELETE]",
    task_description="Milvus garbage collection task",
    non_retryable_exceptions=(LocalServiceUnavailableException,),
)
def delete_embeddings_task(self, **kwargs) -> dict:
    embedding_store = MilvusEmbeddingStore()
    to_time = utc_now()

    for asset_type in settings.AIOD.ASSET_TYPES:
        logging.info(
            f"\t[RECURRING MILVUS DELETE] Deleting embeddings of asset type: {asset_type.value}"
        )
        asyncio.run(delete_asset_embeddings(embedding_store, asset_type, to_time=to_time))

    return {}


@celery_app.task(bind=True, acks_late=False, task_reject_on_worker_lost=False)
@maintenance_task(
    log_prefix="[RECURRING MONGODB DELETE]",
    task_description="MongoDB cleanup task",
    non_retryable_exceptions=(LocalServiceUnavailableException,),
)
def mongo_cleanup_task(self, **kwargs) -> dict:
    asyncio.run(clean_mongo_database())
    return {}


@celery_app.task(bind=True, acks_late=False, task_reject_on_worker_lost=False)
@maintenance_task(
    log_prefix="[RECURRING METADATA EXTRACTION]",
    task_description="metadata extraction task",
    non_retryable_exceptions=(LocalServiceUnavailableException,),
    skip_condition=lambda: (
        not settings.PERFORM_METADATA_EXTRACTION,
        "Metadata extraction disabled",
    ),
)
def extract_metadata_task(self, **kwargs) -> dict:
    asyncio.run(extract_metadata_for_assets())
    return {}


@celery_app.task(bind=True, acks_late=False, task_reject_on_worker_lost=False)
@maintenance_task(
    log_prefix="[RECURRING SCRAPING]",
    task_description="scraping task",
    non_retryable_exceptions=(LocalServiceUnavailableException,),
    skip_condition=lambda: (not settings.CHATBOT.USE_CHATBOT, "Chatbot disabled"),
)
def scraping_task(self, **kwargs) -> dict:
    asyncio.run(populate_collections_wrapper())
    return {}
