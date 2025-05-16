from __future__ import annotations

import threading
from typing import Any, Generic, Type, TypeVar

from tinydb import Query, TinyDB
from tinydb.storages import JSONStorage
from tinydb.table import Table
from tinydb_serialization import SerializationMiddleware, Serializer
from tinydb_serialization.serializers import DateTimeSerializer

from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.db_entity import DatabaseEntity
from app.models.query import FilteredUserQuery, RecommenderUserQuery, SimpleUserQuery
from app.schemas.enums import AssetType, QueryStatus


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

    def decode(self, s) -> QueryStatus:
        return QueryStatus(s)


class AssetTypeSerializer(Serializer):
    OBJ_CLASS = AssetType

    def encode(self, obj) -> str:
        return obj.value

    def decode(self, s) -> AssetType:
        return AssetType(s)


DbEntity = TypeVar("DbEntity", bound=DatabaseEntity)


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

        self.collections: dict[Type[DatabaseEntity], MyCollection] = {
            SimpleUserQuery: MyCollection[SimpleUserQuery](
                self.db.table("simple_queries"),
            ),
            FilteredUserQuery: MyCollection[FilteredUserQuery](
                self.db.table("filtered_queries"),
            ),
            AssetCollection: MyCollection[AssetCollection](
                self.db.table("asset_collections"),
            ),
            RecommenderUserQuery: MyCollection[RecommenderUserQuery](
                self.db.table("recommender_queries"),
            ),
        }

    def insert(self, obj: DatabaseEntity) -> Any:
        with self.db_lock:
            return self.collections[type(obj)].insert(obj)

    def upsert(self, obj: DatabaseEntity) -> Any:
        with self.db_lock:
            return self.collections[type(obj)].upsert(obj)

    def find_by_id(self, type: Type[DbEntity], id: str) -> DbEntity | None:
        with self.db_lock:
            return self.collections[type].find_by_id(type, id)

    def delete(self, type: Type[DbEntity], *args, **kwargs) -> Any:
        with self.db_lock:
            return self.collections[type].delete(*args, **kwargs)

    def search(self, type: Type[DbEntity], *args, **kwargs) -> list[DbEntity]:
        with self.db_lock:
            return self.collections[type].search(type, *args, **kwargs)

    def get_first_asset_collection_by_type(self, asset_type: AssetType) -> AssetCollection | None:
        rs = self.search(AssetCollection, Query().aiod_asset_type == asset_type)
        if len(rs) == 0:
            return None
        return rs[0]


class MyCollection(Generic[DbEntity]):
    def __init__(self, table: Table) -> None:
        self.table = table

    def insert(self, object: DbEntity) -> Any:
        return self.table.insert(object.model_dump())

    def upsert(self, object: DbEntity) -> Any:
        return self.table.upsert(object.model_dump(), Query().id == object.id)

    def find_by_id(self, type: Type[DbEntity], id: str) -> DbEntity | None:
        obj = self.table.get(Query().id == id)
        if obj is None:
            return None
        return type(**obj)

    def delete(self, *args, **kwargs):
        return self.table.remove(*args, **kwargs)

    def search(self, type: Type[DbEntity], *args, **kwargs) -> list[DbEntity]:
        results = self.table.search(*args, **kwargs)
        return [type(**res) for res in results]
