import os
from pymilvus import MilvusClient, DataType
import pandas as pd
from dotenv import load_dotenv


def load_tsv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, sep="\t").set_index("old")


def get_collection_names(csv_collection_names: str, prefix: str) -> list[str]:
    return [f"{prefix}_{s.strip()}" for s in csv_collection_names.split(",")]


def get_collection_fields_to_store(client: MilvusClient, collection_name: str) -> list[str]:
    all_fields = set(
        [field_obj["name"] for field_obj in client.describe_collection(collection_name)["fields"]]
    )
    return list(all_fields - {"id"})


def update_object(object: dict, table: pd.DataFrame) -> dict | None:
    old_id = object["asset_id"]

    try:
        new_id = table.loc[old_id].values[0]
        object["asset_id"] = new_id
        del object["id"]

        return object
    except Exception:
        print("Old ID doesn't exist")
        return None


# Copied from populate_milvus.py script
def create_new_collection(
    client: MilvusClient, collection_name: str, extract_metadata: bool
) -> bool:
    if client.has_collection(collection_name):
        return False

    vector_index_kwargs = {
        "index_type": "HNSW_SQ",
        "metric_type": "COSINE",
        "params": {"sq_type": "SQ8"},
    }
    scalar_index_kwargs = {"index_type": "INVERTED"}

    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)

    # CHANGED DATA TYPE OF ASSET_ID FIELD <<<<<<<<<<<<<<<
    schema.add_field("asset_id", DataType.VARCHAR, max_length=50)

    if extract_metadata and collection_name.endswith("_datasets"):
        # TODO for now this is a duplicate code to the index creation process found
        # in the embedding_store.py file

        # TODO so far it only works with datasets
        # (the same goes for the Milvus index in the main application)
        schema.add_field("date_published", DataType.VARCHAR, max_length=22, nullable=True)
        schema.add_field("size_in_mb", DataType.FLOAT, nullable=True)
        schema.add_field("license", DataType.VARCHAR, max_length=20, nullable=True)

        schema.add_field(
            "task_types",
            DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_length=50,
            max_capacity=100,
            nullable=True,
        )
        schema.add_field(
            "languages",
            DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_length=2,
            max_capacity=200,
            nullable=True,
        )
        schema.add_field("datapoints_upper_bound", DataType.INT64, nullable=True)
        schema.add_field("datapoints_lower_bound", DataType.INT64, nullable=True)

    schema.verify()

    index_params = client.prepare_index_params()

    index_params.add_index(field_name="vector", **vector_index_kwargs)
    index_params.add_index(field_name="asset_id", **scalar_index_kwargs)

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    return True


def update_milvus_collections(
    client: MilvusClient, collection_names: list[str], tsv_conversion_dirpath: str
) -> None:
    tsv_filenames = [f"{name}.tsv" for name in collection_names]
    tsv_tables = [
        load_tsv(os.path.join(tsv_conversion_dirpath, tsv_name)) for tsv_name in tsv_filenames
    ]
    existing_collections = client.list_collections()

    for collection_name, tsv_table in zip(collection_names, tsv_tables):
        if collection_name not in existing_collections:
            print(f"Collection '{collection_name}' doesn't exist...")
            continue

        print(f"===== Migrating asset IDs in the '{collection_name}' Milvus collection =====")

        old_collection_name = f"old_{collection_name}"
        client.rename_collection(collection_name, old_collection_name)
        create_new_collection(client, collection_name, extract_metadata=True)

        field_names = get_collection_fields_to_store(client, old_collection_name)
        iterator = client.query_iterator(
            old_collection_name, batch_size=1000, output_fields=field_names
        )

        while True:
            batch = iterator.next()
            if not batch:
                iterator.close()
                break

            batch = [update_object(b, tsv_table) for b in batch]
            batch = [b for b in batch if b is not None]
            client.insert(collection_name=collection_name, data=batch)

        client.drop_collection(old_collection_name)


if __name__ == "__main__":
    load_dotenv()

    tsv_conversion_dirpath = os.getenv("TSV_DIRPATH", "/data/id_tsv_files")
    milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_token = os.getenv("MILVUS_TOKEN", "root:Milvus")
    milvus_prefix = os.getenv("MILVUS_PREFIX", "AIoD_STS_GTE")

    collection_names = get_collection_names(
        os.getenv("MILVUS_CSV_COLLECTIONS", "datasets"), milvus_prefix
    )
    client = MilvusClient(uri=milvus_uri, token=milvus_token)

    update_milvus_collections(client, collection_names, tsv_conversion_dirpath)
