from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from queue import Queue

from app.config import settings
from app.schemas import SemanticSearchResults
from app.schemas.query import Query, QueryStatus
from app.services.embedding_store import Milvus_EmbeddingStore
from app.services.inference.model import AiModel

QUERY_QUEUE = Queue()


class QueryResultsManager:
    _instance: QueryResultsManager | None = None

    def __new__(cls) -> QueryResultsManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self) -> None:
        self.data: dict[str, Query] = {}

    def add_query(self, query_id: str) -> None:
        self.data[query_id] = Query(id=query_id)

    def get_query(self, query_id: str) -> Query | None:
        query = self.data.get(query_id, None)
        if query is None:
            return None

        if query.status == QueryStatus.COMPLETED:
            self.data[query_id].expires_at = datetime.now(tz=UTC) + timedelta(
                minutes=10
            )
        return self.data[query_id]

    def set_query_status(self, query_id: str, status: QueryStatus) -> bool:
        if self.data.get(query_id, None) is None:
            return False

        self.data[query_id].status = status

    def set_query_result(
        self, query_id: str, query_result: SemanticSearchResults
    ) -> bool:
        if self.data.get(query_id, None) is None:
            return False

        self.set_query_status(query_id, QueryStatus.COMPLETED)
        self.data[query_id].result = query_result
        self.data[query_id].expires_at = datetime.now(tz=UTC) + timedelta(days=1)
        return True


# TODO we need a cron job for cleaning the QueryResultsManager.data variable...


def search_thread() -> None:
    # already instantiated singletons
    model = AiModel()
    vector_store = Milvus_EmbeddingStore()
    query_manager = QueryResultsManager()

    while True:
        queue_item = QUERY_QUEUE.get()
        if queue_item is None:
            return

        query_id = queue_item.id
        query_manager.set_query_status(query_id, QueryStatus.IN_PROGESS)

        results = vector_store.retrieve_topk_document_ids(
            model,
            queue_item,
            collection_name=settings.MILVUS.COLLECTION,
            topk=settings.MILVUS.TOPK,
        )
        query_manager.set_query_result(query_id, results)


def start_search_thread() -> threading.Thread:
    thread = threading.Thread(target=search_thread)
    thread.start()
    return thread
