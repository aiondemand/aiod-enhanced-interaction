from datetime import datetime, timezone
from functools import partial
from uuid import uuid4

from app.config import settings
from app.schemas.enums import AssetType
from pydantic import BaseModel, Field


class CollectionUpdate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    updated_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    aiod_assets_added: int = 0
    finished: bool = False

    def update(self, assets_added: int) -> None:
        self.aiod_assets_added += assets_added
        self.updated_at = datetime.now(tz=timezone.utc)

    def finish(self) -> None:
        self.finished = True
        self.updated_at = datetime.now(tz=timezone.utc)


class SetupCollectionUpdate(CollectionUpdate):
    aiod_asset_offset: int = 0

    def update(self, assets_added: int, **kwargs) -> None:
        self.aiod_asset_offset += settings.AIOD.WINDOW_SIZE
        super().update(assets_added)


class RecurringCollectionUpdate(CollectionUpdate):
    assets_from_time: datetime
    aiod_assets_updated: int = 0

    @property
    def assets_to_time(self) -> datetime:
        return self.created_at

    def update(self, assets_added: int, assets_updated: int = 0) -> None:
        self.aiod_assets_updated(assets_updated)
        super().update(assets_added)


class AssetCollection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    aiod_asset_type: AssetType
    created_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    updated_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    setup_update: SetupCollectionUpdate = SetupCollectionUpdate()
    recurring_updates: list[RecurringCollectionUpdate] = []

    @property
    def setup_done(self) -> bool:
        return self.setup_update.finished

    @property
    def last_update(self) -> SetupCollectionUpdate | RecurringCollectionUpdate:
        return (
            self.setup_update
            if len(self.recurring_updates) == 0
            else self.recurring_updates[-1]
        )

    @property
    def num_assets(self) -> int:
        setup_assets = self.setup_update.aiod_assets_added
        return setup_assets + sum(
            [update.aiod_assets_added for update in self.recurring_updates]
        )

    def add_recurring_update(self) -> None:
        self.recurring_updates.append(
            RecurringCollectionUpdate(assets_from_time=self.last_update.created_at)
        )
        self.updated_at = datetime.now(tz=timezone.utc)

    def update(self, assets_added: int, assets_updated: int = 0) -> None:
        self.last_update.update(
            assets_added=assets_added, assets_updated=assets_updated
        )
        self.updated_at = datetime.now(tz=timezone.utc)

    def finish(self) -> None:
        self.last_update.finish()
        self.updated_at = datetime.now(tz=timezone.utc)
