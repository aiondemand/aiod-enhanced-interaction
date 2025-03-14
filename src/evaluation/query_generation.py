import pandas as pd
import numpy as np
from tqdm import tqdm
import uuid
import json
import os
from typing import Type, Literal
from abc import ABC, abstractmethod
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel, Field
from langchain_core.language_models.llms import BaseLLM

from lang_chains import SimpleChain, LLM_Chain, load_llm
from data_types import AnnotatedDoc, QueryDatapoint


class AssetSpecificQueries(BaseModel):
    least_descriptive: str = Field(..., description="A concise user query, up to 70 characters, capturing only the essential and most significant properties of the dataset.")
    moderately_descriptive: str = Field(..., description="A detailed user query, up to 200 characters, providing additional information and properties to offer a clearer description of the dataset.")
    most_descriptive: str = Field(..., description="A comprehensive user query, up to 500 characters, encompassing a wide range of details and characteristics to thoroughly describe the dataset.")


class GenericQueries(BaseModel):
    least_descriptive: list[str] = Field(..., description="A list of concise user queries, each up to 70 characters, capturing only the essential aspects of the assets we wish to search for.")
    moderately_descriptive: list[str] = Field(..., description="A list of detailed user queries, each up to 200 characters, providing additional details about the assets we wish to search for.")
    most_descriptive: list[str] = Field(..., description="A list of comprehensive user queries each up to 500 characters, encompassing a wide range of details and characteristics of the assets we wish to search for.")


class QueryGeneration(ABC):
    @abstractmethod
    def generate(self, savedir: str, **kwargs) -> None:
        pass

    @abstractmethod
    def get_query_types(self) -> list[str]:
        pass


class AssetSpecificQueryGeneration(QueryGeneration):
    asset_qualities = [
        "long_description_many_tags",
        "long_description_few_tags",
        "moderate_description_many_tags",
        "poor_description_many_tags"
    ]
    descriptiveness_levels = [
        "least_descriptive",
        "moderately_descriptive",
        "most_descriptive"
    ]

    system_prompt = """
        You are an AI language model tasked with generating asset-specific user queries for searching machine learning (ML) assets.
        Your goal is to create detailed and specific user queries that, based on the provided asset description and its metadata,
        would likely retrieve said corresponding asset from the database.

        You will be given a description and additional metadata or keywords describing a particular ML asset.
        This information includes the asset's properties, contents, and other relevant details.

        Based on the input information, you're expected to generate three user queries with varying levels of descriptiveness:
        - Least Descriptive Query: A concise user query, up to 70 characters, capturing only the essential and most significant properties of the asset.
        - Moderately Descriptive Query: A detailed user query, up to 200 characters, providing additional information and properties to offer a clearer description of the asset.
        - Most Descriptive Query: A comprehensive user query, up to 500 characters, encompassing a wide range of details and characteristics to thoroughly describe the asset.

        Important note: You are forbidden to mention any asset names directly in the user queries.

    """
    input_prompt = """
        ### ML asset name to exclude: {name}
        ### ML asset description: {description}
        ### ML asset keywords and tags: {keywords}

        ### User queries to generate:
    """

    @classmethod
    def build_chain(
        cls, llm: BaseLLM | None = None, pydantic_model: Type[BaseModel] | None = None
    ) -> SimpleChain:
        if llm is None:
            llm = load_llm()
        if pydantic_model is None:
            pydantic_model = AssetSpecificQueries
        prompt_templates = [
            cls.system_prompt, cls.input_prompt
        ]

        return LLM_Chain.build_simple_chain(llm, pydantic_model, prompt_templates)

    def __init__(
        self, asset_filepath: str, chain: SimpleChain | None = None,
    ) -> None:
        super().__init__()
        self.asset_filepath = asset_filepath
        with open(asset_filepath) as f:
            self.all_assets = json.load(f)

        self.chain = chain
        if chain is None:
            self.chain = self.build_chain()

    def __call__(self, input: dict) -> dict | str | None:
        return self.chain.invoke(input)

    def generate(self, savedir: str) -> None:
        descr_levels = self.descriptiveness_levels
        all_query_types = self.get_query_types()
        outputs = [[] for _ in range(len(all_query_types))]
        if sum([
            os.path.exists(os.path.join(savedir, f"{qtype}.json"))
            for qtype in all_query_types
        ]) == len(all_query_types):
            return

        with get_openai_callback() as cb:
            for asset_type_it, asset_q in enumerate(self.asset_qualities):
                print(f"...Generating asset-specific queries for '{asset_q}' documents...")
                for doc in tqdm(self.all_assets[asset_q], total=len(self.all_assets[asset_q])):
                    name = doc["name"].split("/")[-1]
                    description = doc["description"]["plain"]
                    keywords = " | ".join(doc["keyword"])

                    output = self({
                        "name": name,
                        "description": description,
                        "keywords": keywords
                    })
                    if output is not None:
                        for descr_it, descr_level in enumerate(descr_levels):
                            out_idx = asset_type_it*len(descr_levels) + descr_it
                            datapoint = QueryDatapoint(
                                text=output[descr_level],
                                id=str(uuid.uuid4()),
                                annotated_docs=[
                                    AnnotatedDoc(id=str(doc["identifier"]), score=1)
                                ]
                            )
                            outputs[out_idx].append(datapoint.model_dump())
                    else:
                        print(f"We were unable to generate asset-specific queries to the asset ID={doc['identifier']}")
            # print(cb)

        os.makedirs(savedir, exist_ok=True)
        for it, qtype in enumerate(all_query_types):
            p = os.path.join(savedir, f"{qtype}.json")
            with open(p, "w") as f:
                json.dump(outputs[it], f, ensure_ascii=False)

    def get_query_types(self) -> list[str]:
        all_query_types = []
        for asset in self.asset_qualities:
            all_query_types.extend([
                f"{level}-{asset}" for level in self.descriptiveness_levels
            ])

        return all_query_types

    @classmethod
    def create_asset_dataset_for_asset_specific_queries(
        cls, json_dirpath: str, savepath: str, max_docs_per_doctype: int = 50
    ) -> None:
        def is_good_dataset(ds: dict) -> bool:
            platform = ds.get("platform", None)
            name = ds.get("name", None)
            description = ds.get("description", {}).get("plain", None)
            keywords = ds.get("keyword", [])

            if (
                name is None or
                description is None or description == "" or
                len(keywords) == 0 or
                platform != "huggingface"
            ):
                return False
            return True

        if os.path.exists(savepath):
            return

        all_good_datasets = []
        ds_ids = []
        descriptions = []
        description_lengths = []
        num_tags = []
        for filename in tqdm(os.listdir(json_dirpath)):
            p = os.path.join(json_dirpath, filename)
            with open(p) as f:
                datasets = json.load(f)

            good_datasets = [ds for ds in datasets if is_good_dataset(ds)]
            all_good_datasets.extend(good_datasets)

            ds_ids.extend([ds["identifier"] for ds in good_datasets])
            descriptions.extend(ds["description"]["plain"] for ds in good_datasets)
            description_lengths.extend(
                [len(ds["description"]["plain"]) for ds in good_datasets]
            )
            num_tags.extend([len(ds["keyword"]) for ds in good_datasets])

        data = pd.DataFrame(data=[ds_ids, descriptions, description_lengths, num_tags]).T
        data.columns = ["id", "description", "descr_len", "num_tags"]
        data = data.set_index("id", drop=True)
        data = data.drop_duplicates(subset=["description"])

        long_descr_many_tags = data[(data["descr_len"] > 1000) & (data["num_tags"] > 10)].iloc[:max_docs_per_doctype].index.values #79 documents
        long_descr_few_tags = data[(data["descr_len"] > 1000) & (data["num_tags"] < 3)].iloc[:max_docs_per_doctype].index.values #95 documents
        moder_descr = data[(data["descr_len"] > 200) & (data["descr_len"] < 500) & (data["num_tags"] > 10)].iloc[:max_docs_per_doctype].index.values #518 documents
        poor_descr = data[(data["descr_len"] < 50) & (data["num_tags"] > 10)].iloc[:max_docs_per_doctype].index.values #61 documents

        all_id_subsets = [
            long_descr_many_tags, long_descr_few_tags, moder_descr, poor_descr
        ]
        json_data = {}
        ds_ids = np.array(ds_ids)
        for doc_ids, asset_q in zip(all_id_subsets, cls.asset_qualities):
            indices = [np.where(ds_ids == doc_id)[0][0] for doc_id in doc_ids]
            selected_datasets = [all_good_datasets[idx] for idx in indices]
            json_data[asset_q] = selected_datasets

        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, "w") as f:
            json.dump(json_data, f, ensure_ascii=False)


class GenericQueryGeneration(QueryGeneration):
    query_types = [
        "least_descriptive",
        "moderately_descriptive",
        "most_descriptive"
    ]

    system_prompt = """
        You are an AI language model tasked with generating generic user queries for searching machine learning (ML) assets,
        primarily those similar to what can be found on HuggingFace. Your goal is to create a diverse set of free-text user queries
        that reflect the types of searches users might perform when looking for such assets, namely {asset}s, on the website.
        These queries should cover a wide range of possible use cases and requirements.

        ### Instructions:

        Create a variety of free-text user queries that represent different possible searches users might perform when searching for ML {asset}s.
        The queries can cover different aspects, such as application, data domain, as well as additional specific asset traits (e.g.,
        task, content, language, {asset} size, framework, format, ...).

        You're expected to generate unique {total_queries} user queries in total with varying levels of descriptiveness:
        - Least Descriptive Queries: {query1} unique queries, each up to 70 characters, capturing only the essential aspects of the assets we wish to search for.
        - Moderately Descriptive Queries: {query2} unique queries, each up to 200 characters, providing additional details in the form at least 2 specific properties of the assets we wish to search for.
        - Most Descriptive Queries: {query3} unique queries, each up to 500 characters, encompassing a wide range of details and characteristics by defining at least 4 specific properties of the assets we wish to search for.

        Important Note: Ensure the queries are realistic and representative of what users might search for. Do not reference any specific asset names directly.
    """
    input_prompt = """
        ### User queries to generate:
    """

    @classmethod
    def build_chain(
        cls, llm: BaseLLM | None = None, pydantic_model: Type[BaseModel] | None = None,
        asset_type: Literal["dataset", "model", "publication"] = "dataset",
        query_counts: list[int] = [10, 30, 50]
    ) -> SimpleChain:
        if llm is None:
            llm = load_llm()
        if pydantic_model is None:
            pydantic_model = GenericQueries
        prompt_templates = [
            cls.system_prompt.format(
                asset=asset_type,
                total_queries=sum(query_counts),
                query1=query_counts[0],
                query2=query_counts[1],
                query3=query_counts[2]
            ),
            cls.input_prompt
        ]

        return LLM_Chain.build_simple_chain(llm, pydantic_model, prompt_templates)

    def __init__(self, chain: SimpleChain | None = None) -> None:
        super().__init__()

        self.chain = chain
        if chain is None:
            self.chain = self.build_chain()

    def __call__(self, input: dict) -> dict | str | None:
        return self.chain.invoke(input)

    def generate(self, savedir: str, num_generative_calls: int = 1) -> None:
        outputs = [[] for _ in range(len(self.query_types))]
        if sum([
            os.path.exists(os.path.join(savedir, f"{qtype}.json"))
            for qtype in self.query_types
        ]) == len(self.query_types):
            return

        print("...Generating generic queries...")
        with get_openai_callback() as cb:
            for _ in tqdm(range(num_generative_calls)):
                output = self({})
                if output is not None:
                    for it, qtype in enumerate(self.query_types):
                        queries = [
                            QueryDatapoint(
                                **{ "id": str(uuid.uuid4()), "text": q }
                            ).model_dump() for q in output[qtype]
                        ]
                        outputs[it].extend(queries)
                else:
                    print("We were unable to generate some generic queries")
            # print(cb)


        os.makedirs(savedir, exist_ok=True)
        for it, qtype in enumerate(self.query_types):
            p = os.path.join(savedir, f"{qtype}.json")
            with open(p, "w") as f:
                json.dump(outputs[it], f, ensure_ascii=False)

    def get_query_types(self) -> list[str]:
        return self.query_types
