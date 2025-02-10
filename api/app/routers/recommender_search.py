from typing import Annotated
from uuid import UUID

from app.models.query import SimilarQuery
from app.schemas.enums import AssetType
from app.schemas.query import SimilarQueryResponse
from app.services.database import Database
from app.services.threads.search_thread import QUERY_QUEUE
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.post("")
async def submit_recommender_query(
    database: Annotated[Database, Depends(Database)],
    asset_id: int = Query(..., description="Asset ID"),
    asset_type: AssetType = Query(..., description="Asset type"),
    topk: int = Query(
        default=10, gt=0, le=100, description="Number of similar assets to return"
    ),
) -> RedirectResponse:
    query_job = SimilarQuery(
        asset_id=asset_id,
        asset_type=asset_type,
        topk=topk,
    )
    database.insert(query_job)
    QUERY_QUEUE.put((query_job.id, type(query_job)))

    return RedirectResponse(url=f"/recommender/{query_job.id}/result", status_code=202)


@router.get("/{query_id}/result", response_model=SimilarQueryResponse)
async def get_recommender_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID,
) -> SimilarQueryResponse:
    query_obj = database.find_by_id(SimilarQuery, id=str(query_id))
    if query_obj is None:
        raise HTTPException(status_code=404, detail="Query not found")
    return query_obj.map_to_response()
