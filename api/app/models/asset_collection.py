from abc import ABC
from datetime import datetime, timezone
from functools import partial
from uuid import uuid4

from pydantic import Field

from app.config import settings
from app.models.db_entity import DatabaseEntity
from app.schemas.enums import AssetType


class CollectionUpdate(DatabaseEntity, ABC):
    embeddings_added: int = 0
    aiod_asset_offset: int = 0
    finished: bool = False

    @property
    def to_time(self) -> datetime:
        return self.created_at

    def update(self, embeddings_added: int, embeddings_removed: int = 0) -> None:
        self.embeddings_added += embeddings_added
        self.aiod_asset_offset += settings.AIOD.OFFSET_INCREMENT
        self.updated_at = datetime.now(tz=timezone.utc)

    def finish(self) -> None:
        self.finished = True
        self.updated_at = datetime.now(tz=timezone.utc)


class SetupCollectionUpdate(CollectionUpdate):
    pass


class RecurringCollectionUpdate(CollectionUpdate):
    from_time: datetime
    embeddings_removed: int = 0

    def update(self, embeddings_added: int, embeddings_removed: int = 0) -> None:
        self.embeddings_removed += embeddings_removed
        super().update(embeddings_added)


class AssetCollection(DatabaseEntity):
    aiod_asset_type: AssetType
    setup_update: SetupCollectionUpdate = SetupCollectionUpdate()
    recurring_updates: list[RecurringCollectionUpdate] = []

    @property
    def setup_done(self) -> bool:
        return self.setup_update.finished

    @property
    def last_update(self) -> SetupCollectionUpdate | RecurringCollectionUpdate:
        return self.setup_update if len(self.recurring_updates) == 0 else self.recurring_updates[-1]

    @property
    def num_assets(self) -> int:
        setup_assets_added = self.setup_update.embeddings_added

        recurr_add = sum([upd.embeddings_added for upd in self.recurring_updates])
        recurr_del = sum([upd.embeddings_removed for upd in self.recurring_updates])

        return setup_assets_added + recurr_add - recurr_del

    def add_recurring_update(self) -> None:
        self.recurring_updates.append(
            RecurringCollectionUpdate(from_time=self.last_update.created_at)
        )
        self.updated_at = datetime.now(tz=timezone.utc)

    def update(self, embeddings_added: int, embeddings_removed: int = 0) -> None:
        self.last_update.update(
            embeddings_added=embeddings_added, embeddings_removed=embeddings_removed
        )
        self.updated_at = datetime.now(tz=timezone.utc)

    def finish(self) -> None:
        self.last_update.finish()
        self.updated_at = datetime.now(tz=timezone.utc)
