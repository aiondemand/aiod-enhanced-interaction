import numpy as np
import os

from dataset import Queries
from embedding_stores import EmbeddingStore
from evaluation.metrics import SpecificAssetQueriesEvaluation
from evaluation.query_generation import AssetSpecificQueryGeneration
from model.lm_encoders.models import EmbeddingModel


class HitRateEvaluationPipeline:
    # TODO retrieval_model should be later or replaced by the retrieval system abstraction
    def __init__(
        self, embedding_model: EmbeddingModel,
        embedding_store: EmbeddingStore,
        model_name: str,
        orig_json_assets_dirpath: str,
        quality_assets_path: str,
        generate_queries_dirpath: str,
        topk_dirpath: str, 
        metrics_dirpath: str,
        retrieve_topk_documents_func_kwargs: dict | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.embedding_store = embedding_store
    
        self.model_name = model_name
        self.orig_json_assets_dirpath = orig_json_assets_dirpath
        self.quality_assets_path = quality_assets_path
        self.generate_queries_dirpath = generate_queries_dirpath
        self.topk_dirpath = topk_dirpath
        self.metrics_dirpath = metrics_dirpath
        
        self.retrieve_topk_documents_func_kwargs = retrieve_topk_documents_func_kwargs
        if retrieve_topk_documents_func_kwargs is None:
            self.retrieve_topk_documents_func_kwargs = {}
    
    def execute(self,topk: int = 100) -> None:
        print(f"=========== ACCURACY EVALUATION FOR '{self.model_name}' STARTED ===========")
        print("===== Filtering assets worth creating specific queries to =====")
        AssetSpecificQueryGeneration.create_asset_dataset_for_asset_specific_queries(
            json_dirpath=self.orig_json_assets_dirpath,
            savepath=self.quality_assets_path,
        )
        
        print("===== Generation of asset-specific queries =====")
        query_generation = AssetSpecificQueryGeneration(asset_filepath=self.quality_assets_path)
        query_generation.generate(savedir=self.generate_queries_dirpath)
        query_types = query_generation.get_query_types()

        for query_type in query_types:
            self._inner_pipeline(query_type, topk=topk)

        print(f"=========== ACCURACY EVALUATION FOR '{self.model_name}' CONCLUDED ===========")
        
    def _inner_pipeline(self, query_type: str, topk: int = 100) -> None:
        print(f"===== Evaluating '{query_type}' queries =====")

        query_filepath = os.path.join(self.generate_queries_dirpath, f"{query_type}.json")
        topk_dirpath = os.path.join(self.topk_dirpath, self.model_name, query_type)
        metrics_savepath = os.path.join(
            self.metrics_dirpath, self.model_name, query_type, "results.json"
        )
        
        print("=== Retreving top K documents to each query ===")
        query_loader = Queries(json_path=query_filepath).build_loader()
        self.embedding_store.retrieve_topk_document_ids(
            self.embedding_model, query_loader, topk=topk, 
            load_dirpath=topk_dirpath, save_dirpath=topk_dirpath,
            **self.retrieve_topk_documents_func_kwargs
        )

        print("=== Computing accuracy metrics ===")
        all_topk = np.array([5, 10, 20, 50, 100], dtype=np.int32)
        all_topk = all_topk[all_topk <= topk].tolist()
        eval = SpecificAssetQueriesEvaluation()
        eval.evaluate(
            self.embedding_model, self.embedding_store, query_loader, 
            topk=all_topk,
            load_topk_docs_dirpath=topk_dirpath,
            metrics_savepath=metrics_savepath,
            retrieve_topk_documents_func_kwargs=self.retrieve_topk_documents_func_kwargs
        )
