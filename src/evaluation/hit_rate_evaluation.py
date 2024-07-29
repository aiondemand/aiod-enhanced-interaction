import numpy as np
import os

from dataset import Queries
from embedding_stores import EmbeddingStore
from evaluation.metrics import SpecificAssetQueriesEvaluation
from evaluation.query_generation import AssetSpecificQueryGeneration
from model.retrieval import RetrievalSystem


class HitRateEvaluationPipeline:
    def __init__(
        self, 
        retrieval_system: RetrievalSystem,
        model_name: str,
        orig_json_assets_dirpath: str,
        quality_assets_path: str,
        generate_queries_dirpath: str,
        topk_dirpath: str, 
        metrics_dirpath: str,
    ) -> None:
        self.retrieval_system = retrieval_system
        
        self.model_name = model_name
        self.orig_json_assets_dirpath = orig_json_assets_dirpath
        self.quality_assets_path = quality_assets_path
        self.generate_queries_dirpath = generate_queries_dirpath
        self.topk_dirpath = topk_dirpath
        self.metrics_dirpath = metrics_dirpath
        
    def execute(self) -> None:
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
            self._inner_pipeline(query_type)

        print(f"=========== ACCURACY EVALUATION FOR '{self.model_name}' CONCLUDED ===========")
        
    def _inner_pipeline(self, query_type: str) -> None:
        print(f"===== Evaluating '{query_type}' queries =====")
        query_filepath = os.path.join(self.generate_queries_dirpath, f"{query_type}.json")
        topk_dirpath = os.path.join(self.topk_dirpath, self.model_name, query_type)
        metrics_savepath = os.path.join(
            self.metrics_dirpath, self.model_name, query_type, "results.json"
        )
        
        print("=== Retreving top K documents to each query ===")
        query_loader = Queries(json_path=query_filepath).build_loader()
        self.retrieval_system(
            query_loader, 
            retrieve_topk_document_ids_func_kwargs={
                "load_dirpath": topk_dirpath,
                "save_dirpath": topk_dirpath
            }
        )
        

        print("=== Computing accuracy metrics ===")
        topk_levels = np.array([5, 10, 20, 50, 100], dtype=np.int32)
        topk_levels = topk_levels[topk_levels <= self.retrieval_system.topk].tolist()
        eval = SpecificAssetQueriesEvaluation()
        eval.evaluate(
            self.retrieval_system,
            query_loader, 
            topk_levels=topk_levels,
            load_topk_docs_dirpath=topk_dirpath,
            metrics_savepath=metrics_savepath,
        )
