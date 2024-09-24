from pydantic import BaseModel


class SearchResults(BaseModel):
    doc_ids: list[str]
    distances: list[float]
