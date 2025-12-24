from celery import Celery
from celery.schedules import crontab

from app.config import settings


class MyCelery(Celery):
    def gen_task_name(self, name: str, module: str) -> str:
        # Remove trailing '.tasks' if present
        if module.endswith(".tasks"):
            module = module[:-6]
        # Only keep the last component of the module path after splitting by '.'
        module_base = module.split(".")[-1]
        return super().gen_task_name(name, module_base)


# Create Celery app instance
celery_app = MyCelery(
    "app",
    broker=settings.CELERY.BROKER_URL,
    backend=settings.CELERY.RESULT_BACKEND_URL,
)

# Configure Celery
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task routing - route tasks to appropriate queues
    task_routes={
        "search.*": {"queue": "search"},
        "maintenance.*": {"queue": "maintenance"},
    },
    worker_prefetch_multiplier=1,  # Prefetch one task at a time for better load balancing
    result_expires=86400,
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks to prevent memory leaks
    worker_disable_rate_limits=False,
    task_default_retry_delay=60,  # 1 minute between retries
)

celery_app.autodiscover_tasks(["app.celery_tasks.search", "app.celery_tasks.maintenance"])

# Configure Beat schedule for recurring tasks (uncomment when ready to use)
celery_app.conf.beat_schedule = {
    # Recurring AIoD updates
    "compute-embeddings-daily": {
        "task": "maintenance.compute_embeddings_task",
        "schedule": crontab(hour=0, minute=0),
        "kwargs": {"first_invocation": False},
        "options": {"queue": "maintenance"},
    },
    # Recurring Milvus embedding cleanup
    "delete-embeddings-monthly": {
        "task": "maintenance.delete_embeddings_task",
        "schedule": crontab(
            day_of_month=str(settings.AIOD.DAY_IN_MONTH_FOR_EMB_CLEANING),
            hour=0,
            minute=0,
        ),
        "options": {"queue": "maintenance"},
    },
    # Recurring MongoDB cleanup
    "mongo-cleanup-daily": {
        "task": "maintenance.mongo_cleanup_task",
        "schedule": crontab(hour=0, minute=0),
        "options": {"queue": "maintenance"},
    },
}

if settings.PERFORM_METADATA_EXTRACTION:
    # Recurring Metadata extraction
    celery_app.conf.beat_schedule["extract-metadata-daily"] = {
        "task": "maintenance.extract_metadata_task",
        "schedule": crontab(hour=2, minute=0),
        "options": {"queue": "maintenance"},
    }

if settings.CHATBOT.USE_CHATBOT:
    # Recurring website crawling
    celery_app.conf.beat_schedule["scraping-daily"] = {
        "task": "maintenance.scraping_task",
        "schedule": crontab(hour=0, minute=0),
        "options": {"queue": "maintenance"},
    }
