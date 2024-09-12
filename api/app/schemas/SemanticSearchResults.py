from pydantic import BaseModel


class SemanticSearchResult(BaseModel):
    query_id: str
    doc_ids: list[str]
    distances: list[float] | None = None
