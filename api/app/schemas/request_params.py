from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class RequestParams(BaseModel):
    offset: int = 0
    limit: int
    from_time: datetime | None = None
    to_time: datetime | None = None

    def new_page(self, offset: int = None, limit: int = None) -> RequestParams:
        new_obj = RequestParams(**self.model_dump())

        if offset is not None:
            new_obj.offset = offset
        if limit is not None:
            new_obj.limit = limit
        return new_obj
