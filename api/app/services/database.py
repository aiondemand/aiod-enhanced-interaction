from __future__ import annotations

import threading
from typing import Any, Generic, Type, TypeVar

from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.query import FilteredUserQuery, SimilarUserQuery, SimpleUserQuery
from app.schemas.enums import AssetType, QueryStatus
from pydantic import BaseModel
from tinydb import Query, TinyDB
from tinydb.storages import JSONStorage
from tinydb.table import Table
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


T = TypeVar("T", bound=BaseModel)


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

        settings.TINYDB_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
        self.db = TinyDB(settings.TINYDB_FILEPATH, storage=serialization)
        self.db_lock = threading.Lock()

        self.collections = {
            SimpleUserQuery: MyCollection[SimpleUserQuery](
                self.db.table("simple_queries"), self.db_lock
            ),
            FilteredUserQuery: MyCollection[FilteredUserQuery](
                self.db.table("filtered_queries"), self.db_lock
            ),
            AssetCollection: MyCollection[AssetCollection](
                self.db.table("asset_collections"), self.db_lock
            ),
            SimilarUserQuery: MyCollection[SimilarUserQuery](
                self.db.table("similar_queries"), self.db_lock
            ),
        }

    def insert(self, obj: BaseModel) -> Any:
        return self.collections[type(obj)].insert(obj)

    def upsert(self, obj: BaseModel) -> Any:
        return self.collections[type(obj)].upsert(obj)

    def find_by_id(self, type: Type[T], id: str) -> T | None:
        return self.collections[type].find_by_id(id)

    def delete(self, type: Type[T], *args, **kwargs) -> Any:
        return self.collections[type].delete(*args, **kwargs)

    def search(self, type: Type[T], *args, **kwargs) -> list[T]:
        return self.collections[type].search(*args, **kwargs)

    def get_first_asset_collection_by_type(
        self, asset_type: AssetType
    ) -> AssetCollection | None:
        rs = self.search(AssetCollection, Query().aiod_asset_type == asset_type)
        if len(rs) == 0:
            return None
        return rs[0]


class MyCollection(Generic[T]):
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

    def delete(self, *args, **kwargs):
        with self.db_lock:
            return self.table.remove(*args, **kwargs)

    def search(self, *args, **kwargs) -> list[T]:
        results = self.table.search(*args, **kwargs)
        T_type = self.__orig_class__.__args__[0]
        return [T_type(**res) for res in results]
