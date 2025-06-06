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


# TODO later we may wish to make Document classes (that have their respective collections in db)
# inherit from this class, rather than using its staticmethods
class MongoDocument:
    @retry_loop(MongoUnavailableException)
    @staticmethod
    async def get(type: type[Document], *args, **kwargs) -> Document | None:
        return await type.get(*args, **kwargs)

    @retry_loop(MongoUnavailableException)
    @staticmethod
    async def find_first_or_none(type: type[Document], *args, **kwargs) -> Document | None:
        return await type.find(*args, **kwargs).first_or_none()

    @retry_loop(MongoUnavailableException)
    @staticmethod
    async def find_all(type: type[Document], *args, **kwargs) -> list[Document]:
        return await type.find(*args, **kwargs).to_list()

    @retry_loop(MongoUnavailableException)
    @staticmethod
    async def create(obj: Document, *args, **kwargs) -> Document:
        return await obj.create(*args, **kwargs)

    @retry_loop(MongoUnavailableException)
    @staticmethod
    async def replace(obj: Document, *args, **kwargs) -> Document:
        return await obj.replace(*args, **kwargs)

    @retry_loop(MongoUnavailableException)
    @staticmethod
    async def delete(type: type[Document], *args, **kwargs) -> DeleteResult:
        return await type.find(*args, **kwargs).delete()
