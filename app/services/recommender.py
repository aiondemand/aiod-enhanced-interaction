import logging

from app.models.query import RecommenderUserQuery
from app.services.aiod import get_aiod_asset
from app.services.embedding_store import EmbeddingStore
from app.services.inference.model import AiModel
from app.services.inference.text_operations import ConvertJsonToString


def get_precomputed_embeddings_for_recommender(
    model: AiModel,
    embedding_store: EmbeddingStore,
    user_query: RecommenderUserQuery,
) -> list[list[float]] | None:
    precomputed_embeddings = embedding_store.get_asset_embeddings(
        user_query.asset_id, user_query.asset_type
    )
    if precomputed_embeddings is None:
        logging.warning(
            f"No embedding found for asset_id='{user_query.asset_id}' ({user_query.asset_type.value}) in Milvus."
        )
        asset_obj = get_aiod_asset(user_query.asset_id, user_query.asset_type)
        if asset_obj is None:
            # TODO we should pass the information to the user that the asset_id they provided is invalid
            # For now, current implementation returns an empty list of similar assets to a non-existing asset
            logging.error(
                f"Asset with id '{user_query.asset_id}' ({user_query.asset_type.value}) not found in AIoD platform."
            )
            return None
        stringified_asset = ConvertJsonToString.stringify(asset_obj)
        emb = model.compute_asset_embeddings(stringified_asset)[0]
        precomputed_embeddings = emb.cpu().numpy().tolist()
    return precomputed_embeddings
