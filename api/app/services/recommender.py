import logging

from app.models.query import RecommenderUserQuery
from app.services.aiod import get_aiod_document
from app.services.embedding_store import EmbeddingStore
from app.services.inference.model import AiModel
from app.services.inference.text_operations import ConvertJsonToString


def get_precomputed_embeddings_for_recommender(
    model: AiModel,
    embedding_store: EmbeddingStore,
    user_query: RecommenderUserQuery,
    doc_ids_to_exclude_from_search: list[str],
) -> list | None:
    doc_ids_to_exclude_from_search.append(str(user_query.asset_id))

    precomputed_embeddings = embedding_store.get_asset_embeddings(
        user_query.asset_id, user_query.asset_type
    )
    if not precomputed_embeddings:
        logging.warning(
            f"No embedding found for doc_id='{user_query.asset_id}' in Milvus."
        )
        asset_obj = get_aiod_document(user_query.asset_id, user_query.asset_type)
        if asset_obj is None:
            logging.error(
                f"Asset with id '{user_query.asset_id}' not found in AIoD platform."
            )
            return None
        stringified_asset = ConvertJsonToString.stringify(asset_obj)
        emb = model.compute_asset_embeddings(stringified_asset)[0]
        precomputed_embeddings = emb.cpu().numpy().tolist()
    return precomputed_embeddings
