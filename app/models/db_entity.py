from abc import ABC
from datetime import datetime
from pydantic import BaseModel, Field

from app.services.helper import utc_now


class BaseDatabaseEntity(BaseModel, ABC):
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
