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

        # all the query types separately
        for query_type in query_types:
            self._inner_pipeline(
                query_type_list=[query_type], 
                query_type_name=query_type
            )

        # query types grouped by query descriptiveness level
        descr_levels = query_generation.descriptiveness_levels
        for descr_lvl in descr_levels:
            query_types_subset = [
                q_type for q_type in query_types 
                if q_type.startswith(descr_lvl)
            ]
            self._inner_pipeline(
                query_type_list=query_types_subset, 
                query_type_name=descr_lvl
            )
            
        # query types grouped by asset quality
        asset_qualities = query_generation.asset_qualities
        for asset_quality in asset_qualities:
            query_types_subset = [
                q_type for q_type in query_types 
                if q_type.endswith(asset_quality)
            ]
            self._inner_pipeline(
                query_type_list=query_types_subset, 
                query_type_name=asset_quality
            )

        # all the query types grouped together
        self._inner_pipeline(
            query_type_list=query_types, 
            query_type_name="all"
        )

        print(f"=========== ACCURACY EVALUATION FOR '{self.model_name}' CONCLUDED ===========")
        
    def _inner_pipeline(self, query_type_list: list[str], query_type_name: str) -> None:
        print(f"===== Evaluating '{query_type_name}' queries =====")
        query_filepaths = [
            os.path.join(self.generate_queries_dirpath, f"{path}.json") 
            for path in query_type_list
        ]
        load_topk_dirpaths = [
            os.path.join(self.topk_dirpath, self.model_name, path)
            for path in query_type_list
        ]
        save_topk_dirpath = (
            os.path.join(self.topk_dirpath, self.model_name, query_type_name)
            if len(query_type_list) == 1
            else None
        )
        metrics_savepath = os.path.join(
            self.metrics_dirpath, self.model_name, query_type_name, "results.json"
        )
            
        query_loader = Queries(json_paths=query_filepaths).build_loader()
        
        if save_topk_dirpath is not None:
            print("=== Retreving top K documents to each query ===")
            self.retrieval_system(
                query_loader, 
                retrieve_topk_document_ids_func_kwargs={
                    "load_dirpaths": load_topk_dirpaths,
                    "save_dirpath": save_topk_dirpath
                }
            )
        return #TODO
    
        print("=== Computing accuracy metrics ===")
        topk_levels = np.array([5, 10, 20, 30], dtype=np.int32)
        topk_levels = topk_levels[topk_levels <= self.retrieval_system.topk].tolist()
        eval = SpecificAssetQueriesEvaluation()
        eval.evaluate(
            self.retrieval_system,
            query_loader, 
            topk_levels=topk_levels,
            load_topk_docs_dirpath=load_topk_dirpaths,
            metrics_savepath=metrics_savepath,
        )
