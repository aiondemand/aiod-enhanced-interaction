import logging
from functools import wraps
from typing import Callable, Optional, Tuple, Type, Any

from app.celery_tasks.maintenance.redis_lock import RedisTaskLock


def maintenance_task(
    log_prefix: str,
    task_description: str,
    non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    cleanup_func: Optional[Callable[[Any], None]] = None,
    skip_condition: Optional[Callable[[], Tuple[bool, str]]] = None,
    start_message_func: Optional[Callable[[dict], str]] = None,
    enable_redis_lock: bool = True,
):
    """
    Base decorator for maintenance tasks that handles common patterns:
    - Redis distributed locking (prevents concurrent execution)
    - Logging start/end messages
    - Exception handling (with selective retry)
    - Cleanup operations
    - Skip conditions
    - Return status dictionaries

    Args:
        log_prefix: Prefix for log messages (e.g., "[RECURRING AIOD UPDATE]")
        task_description: Description of the task (e.g., "embedding task")
        non_retryable_exceptions: Tuple of exception types that should NOT be retried (all others will be)
        cleanup_func: Optional cleanup function to run in finally block
        skip_condition: Optional function that returns (should_skip, reason) tuple
        start_message_func: Optional function to generate custom start message based on kwargs
        enable_redis_lock: Whether to use Redis distributed lock (default: True)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> dict:
            # Check skip condition
            if skip_condition:
                should_skip, reason = skip_condition()
                if should_skip:
                    return {"status": "skipped", "reason": reason}

            # Initialize Redis lock
            redis_lock: Optional[RedisTaskLock] = None
            if enable_redis_lock:
                redis_lock = RedisTaskLock(func.__name__)
                # Try to acquire the lock
                if not redis_lock.acquire():
                    return {
                        "status": "skipped",
                        "reason": "Another instance of this task is already running",
                    }

            # Ensure context exists in kwargs (create if not provided)
            if "context" not in kwargs:
                kwargs["context"] = {}
            context = kwargs["context"]

            try:
                # Generate start message
                if start_message_func:
                    start_msg = f"{log_prefix} {start_message_func(kwargs)}"
                else:
                    start_msg = f"{log_prefix} Scheduled task for {task_description} has started."
                logging.info(start_msg)

                # Execute main function - context is now in kwargs
                result = func(*args, **kwargs)

                # Log end message
                end_msg = f"{log_prefix} Scheduled task for {task_description} has ended."
                logging.info(end_msg)

                # Return success status with any additional data from result
                status_dict = {"status": "completed"}
                if isinstance(result, dict):
                    status_dict.update(result)
                return status_dict

            except Exception as e:
                # Check if this exception should NOT be retried
                if non_retryable_exceptions and isinstance(e, non_retryable_exceptions):
                    # Log non-retryable exceptions and return error status
                    logging.error(e)
                    logging.error(
                        f"{log_prefix} The above error has been encountered in the {task_description}."
                    )
                    return {"status": "failed", "error": str(e)}
                else:
                    # Re-raise all other exceptions to trigger Celery retry
                    logging.error(e)
                    logging.error(
                        f"{log_prefix} The above error has been encountered in the {task_description}. "
                        "Task will be retried."
                    )
                    task_self = args[0]
                    raise task_self.retry(exc=e)  # Re-raise to trigger Celery retry
            finally:
                # Run cleanup if provided
                if cleanup_func:
                    cleanup_func(context)

                # Release Redis lock
                if redis_lock:
                    redis_lock.release()

        return wrapper

    return decorator
