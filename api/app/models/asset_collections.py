from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from uuid import uuid4

from app.schemas.enums import AssetType
from pydantic import BaseModel, Field


class AssetCollection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    asset_type: AssetType
    setup_done: bool = False
    num_assets: int = 0
    created_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    updated_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    updates: list[AssetUpdate]


class AssetUpdate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    assets_added: int = 0
    assets_updated: int = 0
    finished: bool = False
