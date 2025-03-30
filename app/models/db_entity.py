from abc import ABC
from datetime import datetime, timezone
from functools import partial
from uuid import uuid4

from pydantic import BaseModel, Field


class DatabaseEntity(BaseModel, ABC):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    updated_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
