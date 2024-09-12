from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class QueryStatus(Enum):
    QUEUED = "Queued"
    IN_PROGESS = "In_progress"
    COMPLETED = "Completed"


class Query(BaseModel):
    id: str
    status: QueryStatus = QueryStatus.QUEUED
    result: list[str] | None = None
    expires_at: datetime | None = None


class QueueItem(BaseModel):
    id: str
    query: str
