from __future__ import annotations

import threading
from typing import Any, Generic, TypeVar

from app.config import settings
from app.models.asset_collections import AssetCollection
from app.models.query import UserQuery
from app.schemas.enums import AssetType, QueryStatus
from pydantic import BaseModel
from tinydb import Query, TinyDB
from tinydb.storages import JSONStorage
from tinydb_serialization import SerializationMiddleware, Serializer
from tinydb_serialization.serializers import DateTimeSerializer


class NoneSerializer(Serializer):
    OBJ_CLASS = type(None)

    def encode(self, obj) -> str:
        return "null"

    def decode(self, s) -> None:
        if s == "null":
            return None
        raise ValueError(f"Unknown serialization value: {s}")


class QueryStatusSerializer(Serializer):
    OBJ_CLASS = QueryStatus

    def encode(self, obj) -> str:
        return obj.value

    def decode(self, s) -> None:
        return QueryStatus(s)


class AssetTypeSerializer(Serializer):
    OBJ_CLASS = AssetType

    def encode(self, obj) -> str:
        return obj.value

    def decode(self, s) -> None:
        return AssetType(s)


class Database:
    _instance: Database | None = None

    def __new__(cls) -> Database:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self) -> None:
        serialization = SerializationMiddleware(JSONStorage)
        serialization.register_serializer(DateTimeSerializer(), "TinyDate")
        serialization.register_serializer(NoneSerializer(), "TinyNone")
        serialization.register_serializer(QueryStatusSerializer(), "TinyQueryStatus")
        serialization.register_serializer(AssetTypeSerializer(), "TinyAssetType")

        self.db = TinyDB(settings.TINYDB_FILEPATH, storage=serialization)
        self.db_lock = threading.Lock()

        self.queries = Table[UserQuery](self.db.table("queries"), self.db_lock)
        self.asset_collections = Table[AssetCollection](
            self.db.table("asset_collections"), self.db_lock
        )


T = TypeVar("T", bound=BaseModel)


class Table(Generic[T]):
    def __init__(self, table: Table, db_lock: threading.Lock) -> None:
        self.table = table
        self.db_lock = db_lock

    def insert(self, object: T) -> Any:
        with self.db_lock:
            return self.table.insert(object.model_dump())

    def upsert(self, object: T) -> Any:
        with self.db_lock:
            return self.table.upsert(object.model_dump(), Query().id == object.id)

    def find_by_id(self, id: str) -> T | None:
        obj = self.table.get(Query().id == id)
        if obj is None:
            return None
        T_type = self.__orig_class__.__args__[0]
        return T_type(**obj)

    def search(self, *args, **kwargs) -> list[T]:
        results = self.table.search(*args, **kwargs)
        T_type = self.__orig_class__.__args__[0]
        return [T_type(**res) for res in results]
