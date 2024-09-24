from __future__ import annotations

import threading
from typing import Any

from app.config import settings
from app.models.query import UserQuery
from app.schemas.query_status import QueryStatus
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


class UserQueryDatabase:
    _instance: UserQueryDatabase | None = None

    def __new__(cls) -> UserQueryDatabase:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self) -> None:
        serialization = SerializationMiddleware(JSONStorage)
        serialization.register_serializer(DateTimeSerializer(), "TinyDate")
        serialization.register_serializer(NoneSerializer(), "TinyNone")
        serialization.register_serializer(QueryStatusSerializer(), "TinyQueryStatus")

        self.db = TinyDB(settings.DB_FILEPATH, storage=serialization)
        self.db_lock = threading.Lock()

    def insert(self, userQuery: UserQuery) -> Any:
        with self.db_lock:
            return self.db.insert(userQuery.model_dump())

    def upsert(self, userQuery: UserQuery) -> Any:
        with self.db_lock:
            return self.db.upsert(userQuery.model_dump(), Query().id == userQuery.id)

    def find_by_id(self, query_id: str) -> UserQuery | None:
        obj = self.db.get(Query().id == query_id)
        if obj is None:
            return None
        return UserQuery(**obj)

    def search(self, *args, **kwargs) -> list[UserQuery]:
        results = self.db.search(*args, **kwargs)
        return [UserQuery(**res) for res in results]
