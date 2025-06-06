from __future__ import annotations

from abc import ABC
from datetime import datetime
from beanie import Document
from pydantic import BaseModel, Field
from pymongo.results import DeleteResult
from app.services.helper import utc_now
from app.services.resilience import MongoUnavailableException, retry_loop


class BaseDatabaseEntity(BaseModel, ABC):
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MongoDocument(Document):
    @classmethod
    @retry_loop(MongoUnavailableException)
    async def get_doc_by_id(cls, *args, **kwargs) -> MongoDocument | None:
        return await cls.get(*args, **kwargs)

    @classmethod
    @retry_loop(MongoUnavailableException)
    async def find_first_doc_or_none(cls, *args, **kwargs) -> MongoDocument | None:
        return await cls.find(*args, **kwargs).first_or_none()

    @classmethod
    @retry_loop(MongoUnavailableException)
    async def find_all_docs(cls, *args, **kwargs) -> list[MongoDocument]:
        return await cls.find(*args, **kwargs).to_list()

    @retry_loop(MongoUnavailableException)
    async def create_doc(self, *args, **kwargs) -> MongoDocument:
        return await self.create(*args, **kwargs)

    @retry_loop(MongoUnavailableException)
    async def replace_doc(self, *args, **kwargs) -> MongoDocument:
        return await self.replace(*args, **kwargs)

    @retry_loop(MongoUnavailableException)
    async def delete_doc(self, *args, **kwargs) -> DeleteResult:
        return await self.find(*args, **kwargs).delete()
