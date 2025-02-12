from app.schemas.search_results import SearchResults


# This function is obsolete now :)
def combine_search_results(
    results_list: list[SearchResults], topk: int = None
) -> SearchResults:

    all_docs = []
    for sr in results_list:
        for doc_id, dist in zip(sr.doc_ids, sr.distances):
            all_docs.append({"doc_id": doc_id, "distance": dist})

    doc_map = {}
    for doc in all_docs:
        d_id = doc["doc_id"]
        d_dist = doc["distance"]
        if d_id not in doc_map or d_dist < doc_map[d_id]["distance"]:
            doc_map[d_id] = doc

    sorted_docs = sorted(doc_map.values(), key=lambda x: x["distance"], reverse=False)

    if topk is not None:
        sorted_docs = sorted_docs[:topk]

    final_doc_ids = [doc["doc_id"] for doc in sorted_docs]
    final_distances = [doc["distance"] for doc in sorted_docs]
    return SearchResults(doc_ids=final_doc_ids, distances=final_distances)
