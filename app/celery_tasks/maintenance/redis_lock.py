"""Redis distributed lock with automatic renewal for maintenance tasks."""

import logging
import threading
from typing import Optional

import redis
from redis.exceptions import RedisError

from app import settings


class RedisTaskLock:
    """
    Distributed lock using Redis with automatic TTL renewal.

    Ensures only one instance of a task can run across multiple workers.
    Automatically renews the lock every RENEWAL_INTERVAL to prevent expiration
    during long-running tasks.
    """

    LOCK_KEY_PREFIX = "celery:maintenance:lock:"

    def __init__(self, task_name: str):
        """
        Initialize Redis lock for a specific task.

        Args:
            task_name: Unique identifier for the task (e.g., 'compute_embeddings_task')
        """
        self.task_name = task_name
        self.lock_key = f"{self.LOCK_KEY_PREFIX}{task_name}"
        self._redis_client: Optional[redis.Redis] = None
        self._renewal_thread: Optional[threading.Thread] = None
        self._stop_renewal = threading.Event()
        self._lock_acquired = False

    @property
    def redis_client(self) -> redis.Redis:
        """Lazy initialization of Redis client."""
        if self._redis_client is None:
            # Parse Redis URL from Celery result backend
            self._redis_client = redis.from_url(
                settings.CELERY.RESULT_BACKEND_URL, decode_responses=True
            )
        return self._redis_client

    def acquire(self) -> bool:
        """
        Attempt to acquire the lock.

        Returns:
            bool: True if lock was acquired, False if already held by another process
        """
        try:
            # Use SET with NX (only set if not exists) and EX (expiration)
            acquired = self.redis_client.set(
                self.lock_key,
                "locked",
                nx=True,  # Only set if key doesn't exist
                ex=settings.CELERY.REDIS_LOCK_TTL_SECONDS,
            )

            if acquired:
                self._lock_acquired = True
                logging.info(
                    f"[REDIS LOCK] Acquired lock for task '{self.task_name}' "
                    f"with {settings.CELERY.REDIS_LOCK_TTL_SECONDS}s TTL"
                )
                # Start automatic renewal thread
                self._start_renewal_thread()
                return True
            else:
                # Check how much time is left on the existing lock
                ttl = self.redis_client.ttl(self.lock_key)
                logging.warning(
                    f"[REDIS LOCK] Failed to acquire lock for task '{self.task_name}'. "
                    f"Lock is held by another process (TTL: {ttl}s remaining)"
                )
                return False

        except RedisError as e:
            logging.error(
                f"[REDIS LOCK] Redis error while acquiring lock for '{self.task_name}': {e}"
            )
            return False

    def release(self) -> None:
        """Release the lock and stop the renewal thread."""
        if not self._lock_acquired:
            return

        # Stop renewal thread first
        self._stop_renewal_thread()

        try:
            deleted = self.redis_client.delete(self.lock_key)
            if deleted:
                logging.info(f"[REDIS LOCK] Released lock for task '{self.task_name}'")
            else:
                logging.warning(
                    f"[REDIS LOCK] Lock for task '{self.task_name}' was already released or expired"
                )
        except RedisError as e:
            logging.error(
                f"[REDIS LOCK] Redis error while releasing lock for '{self.task_name}': {e}"
            )
        finally:
            self._lock_acquired = False
            # Close Redis connection
            if self._redis_client:
                try:
                    self._redis_client.close()
                except Exception:
                    pass
                self._redis_client = None

    def _start_renewal_thread(self) -> None:
        """Start background thread to automatically renew the lock."""
        self._stop_renewal.clear()
        self._renewal_thread = threading.Thread(
            target=self._renewal_loop, name=f"RedisLockRenewal-{self.task_name}", daemon=True
        )
        self._renewal_thread.start()
        logging.info(
            f"[REDIS LOCK] Started renewal thread for task '{self.task_name}' "
            f"(renews every {settings.CELERY.RENEWAL_INTERVAL_SECONDS}s)"
        )

    def _stop_renewal_thread(self) -> None:
        """Stop the renewal thread."""
        if self._renewal_thread and self._renewal_thread.is_alive():
            self._stop_renewal.set()
            self._renewal_thread.join(timeout=5)
            logging.info(f"[REDIS LOCK] Stopped renewal thread for task '{self.task_name}'")

    def _renewal_loop(self) -> None:
        """Background loop that renews the lock every RENEWAL_INTERVAL."""
        while not self._stop_renewal.wait(timeout=settings.CELERY.RENEWAL_INTERVAL_SECONDS):
            try:
                # Check if lock still exists
                exists = self.redis_client.exists(self.lock_key)
                if not exists:
                    logging.warning(
                        f"[REDIS LOCK] Lock for task '{self.task_name}' expired before renewal. "
                        "This should not happen - task may be taking too long."
                    )
                    break

                # Extend the TTL by setting expiration again
                self.redis_client.expire(self.lock_key, settings.CELERY.REDIS_LOCK_TTL_SECONDS)
                logging.info(
                    f"[REDIS LOCK] Renewed lock for task '{self.task_name}' "
                    f"(extended TTL by {settings.CELERY.REDIS_LOCK_TTL_SECONDS}s)"
                )

            except RedisError as e:
                logging.error(
                    f"[REDIS LOCK] Redis error during renewal for '{self.task_name}': {e}"
                )
                # Continue trying to renew despite errors
            except Exception as e:
                logging.error(
                    f"[REDIS LOCK] Unexpected error during renewal for '{self.task_name}': {e}"
                )
                break
