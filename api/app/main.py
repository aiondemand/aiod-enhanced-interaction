import threading

from app.routers import query as query_router
from app.services.database import UserQueryDatabase
from app.services.embedding_store import Milvus_EmbeddingStore
from app.services.inference.model import AiModelForBatchProcessing
from app.services.threads.search_thread import QUERY_QUEUE, start_search_thread
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

batch_thread: threading.Thread | None = None
query_thread: threading.Thread | None = None

app = FastAPI(title="[AIoD] Semantic Search")

app.include_router(query_router.router, prefix="/query", tags=["query"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# TODO replace the deprecated methods
@app.on_event("startup")
async def app_init() -> None:
    instantiate_singletons()

    # global batch_process
    # batch_process = start_embedding_process()

    global query_thread
    query_thread = start_search_thread()
    pass


@app.on_event("shutdown")
async def app_shutdown() -> None:
    QUERY_QUEUE.put(None)
    query_thread.join(timeout=5)
    # batch_process.join(timeout=5)


def instantiate_singletons() -> None:
    # Instantiate all the singletons before utilizing them in other threads
    Milvus_EmbeddingStore()
    UserQueryDatabase()

    # TODO I suppose these models should be loaded in their respective threads
    AiModelForBatchProcessing()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="0.0.0.0")
