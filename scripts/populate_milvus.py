from time import time

import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from pymilvus import DataType, MilvusClient, CollectionSchema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Valid asset type names
VALID_ASSET_TYPES = [
    "datasets",
    "ml_models",
    "publications",
    "case_studies",
    "educational_resources",
    "experiments",
    "services",
]


class InputArgs(BaseModel):
    input_dirpath: str
    uri: str
    username: str
    password: str
    metadata_fields_path: str
    old_collection_prefix: str = "AIoD_STS_GTE"  # Prefix used in existing data/collections
    new_collection_prefix: str = "aiod"  # Prefix to use for creating new collections


def load_metadata_field_config(config_path: str) -> dict:
    """Load metadata field configuration from JSON file."""
    config_file_path = Path(config_path)
    if not config_file_path.exists():
        raise ValueError(f"Incorrect path to the Milvus Metadata Fields file: {config_path}")
    with config_file_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config


def resolve_data_type(data_type_name: str) -> DataType:
    """Convert string data type name to DataType enum."""
    try:
        return getattr(DataType, data_type_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported Milvus data type: {data_type_name}") from exc


def add_metadata_field(schema: CollectionSchema, field_definition: dict[str, Any]) -> None:
    """Add a single metadata field to the schema."""
    if "field_name" not in field_definition or "data_type" not in field_definition:
        raise ValueError("Field configuration must contain 'field_name' and 'data_type'")

    field_name = field_definition["field_name"]
    data_type = resolve_data_type(field_definition["data_type"])

    field_kwargs: dict[str, Any] = {
        key: value
        for key, value in field_definition.items()
        if key not in {"field_name", "data_type", "element_type"}
    }

    if "element_type" in field_definition:
        field_kwargs["element_type"] = resolve_data_type(field_definition["element_type"])

    schema.add_field(field_name, data_type, **field_kwargs)


def metadata_fields_for(asset_type: str, metadata_config: dict) -> list[dict[str, Any]]:
    """Get metadata fields for a specific asset type (base + type-specific)."""
    base_fields = metadata_config.get("base", [])
    type_fields = metadata_config.get(asset_type, [])
    return [*base_fields, *type_fields]


def add_metadata_fields(schema: CollectionSchema, asset_type: str, metadata_config: dict) -> None:
    """Add all metadata fields for an asset type to the schema."""
    for field in metadata_fields_for(asset_type, metadata_config):
        try:
            add_metadata_field(schema, field)
        except Exception as exc:
            logger.warning(
                f"Failed to register Milvus metadata field '{field.get('field_name')}' "
                f"for asset type '{asset_type}': {exc}"
            )


def get_collection_name(prefix: str, asset_name: str) -> str:
    """
    Create a Milvus collection name from a prefix and asset name.

    Args:
        prefix: Collection prefix (e.g., "AIoD_STS_GTE")
        asset_name: Asset type name (e.g., "datasets", "ml_models", etc.)

    Returns:
        Collection name in format: {prefix}_{asset_name}

    Raises:
        ValueError: If asset_name is not a valid asset type
    """
    if asset_name not in VALID_ASSET_TYPES:
        raise ValueError(
            f"Invalid asset name: {asset_name}. "
            f"Valid asset types are: {', '.join(VALID_ASSET_TYPES)}"
        )

    return f"{prefix}_{asset_name}"


def extract_asset_type_from_collection_name(
    collection_name: str, collection_prefix: str | None = None
) -> str:
    """
    Extract asset type from collection name.

    Args:
        collection_name: Full collection name (e.g., "AIoD_STS_GTE_datasets")
        collection_prefix: Optional prefix to strip (e.g., "AIoD_STS_GTE")

    Returns:
        Asset type string (e.g., "datasets")

    Raises:
        ValueError: If asset type cannot be extracted or is invalid
    """
    if collection_prefix:
        # Remove prefix and underscore
        if collection_name.startswith(collection_prefix + "_"):
            asset_type = collection_name[len(collection_prefix) + 1 :]
        else:
            raise ValueError(
                f"Collection name '{collection_name}' does not start with prefix '{collection_prefix}'"
            )
    else:
        # Try to extract by finding the last underscore-separated part
        parts = collection_name.split("_")
        # Check if any part matches a valid asset type
        asset_type = None
        for part in reversed(parts):
            if part in VALID_ASSET_TYPES:
                asset_type = part
                break

        if asset_type is None:
            # Fallback: use the last part
            asset_type = parts[-1]

    if asset_type not in VALID_ASSET_TYPES:
        raise ValueError(
            f"Invalid asset type '{asset_type}' extracted from collection name '{collection_name}'. "
            f"Valid asset types are: {', '.join(VALID_ASSET_TYPES)}"
        )

    return asset_type


def populate_database(args: InputArgs) -> None:
    if os.path.exists(args.input_dirpath) is False:
        exit(1)

    client = MilvusClient(uri=args.uri, user=args.username, password=args.password)

    # Load metadata configuration
    metadata_config = load_metadata_field_config(args.metadata_fields_path)

    for old_collection_name in sorted(os.listdir(args.input_dirpath)):
        path = os.path.join(args.input_dirpath, old_collection_name)
        # Skip files, only process directories
        if not os.path.isdir(path):
            continue

        # Extract asset type from old collection name
        try:
            asset_type = extract_asset_type_from_collection_name(
                old_collection_name, args.old_collection_prefix
            )
        except ValueError as e:
            logger.warning(f"Could not extract asset type from '{old_collection_name}': {e}")
            continue

        # Create new collection name with new prefix
        new_collection_name = get_collection_name(args.new_collection_prefix, asset_type)

        populate_collection(
            client,
            old_collection_name,
            new_collection_name,
            path,
            metadata_config,
            args.new_collection_prefix,
        )


def populate_collection(
    client: MilvusClient,
    old_collection_name: str,
    new_collection_name: str,
    data_dirpath: str,
    metadata_config: dict,
    collection_prefix: str | None = None,
) -> None:
    create_new_collection(client, new_collection_name, metadata_config, collection_prefix)

    logger.info(f"Populating collection: {new_collection_name} (from data: {old_collection_name})")

    # Find all .npz files matching page_*.npz pattern
    npz_files = sorted(
        [f for f in os.listdir(data_dirpath) if f.endswith(".npz") and f.startswith("page_")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    if len(npz_files) == 0:
        logger.warning(f"No .npz files found in {data_dirpath}")
        return

    total_pages = len(npz_files)
    logger.info(f"Found {total_pages} page files to process")

    for page_idx, npz_file in enumerate(npz_files, start=1):
        npz_path = os.path.join(data_dirpath, npz_file)

        # Find corresponding JSON file
        json_file = npz_file.replace(".npz", ".json")
        json_path = os.path.join(data_dirpath, json_file)

        if not os.path.exists(json_path):
            logger.warning(
                f"JSON file not found for {npz_file}, skipping page {page_idx}/{total_pages}"
            )
            continue

        logger.info(f"Processing page {page_idx}/{total_pages}: {npz_file}")

        # Load vectors from .npz file
        try:
            npz_data = np.load(npz_path)
            vectors = npz_data["vectors"]  # 2D array: num_docs x vector_dim
        except Exception as e:
            logger.error(f"Error loading vectors from {npz_file}: {e}")
            continue

        # Load asset_ids from .json file
        try:
            with open(json_path) as f:
                asset_ids = json.load(f)
        except Exception as e:
            logger.error(f"Error loading asset_ids from {json_file}: {e}")
            continue

        # Verify vectors and asset_ids have matching lengths
        if len(vectors) != len(asset_ids):
            logger.warning(
                f"Mismatch between vectors ({len(vectors)}) and "
                f"asset_ids ({len(asset_ids)}) in {npz_file}, skipping page {page_idx}/{total_pages}"
            )
            continue

        # Create records
        records = [
            {
                "vector": vec.tolist(),  # Convert numpy array to list
                "asset_id": str(asset_id),
                "asset_version": 0,
            }
            for vec, asset_id in zip(vectors, asset_ids)
        ]

        if len(records) == 0:
            logger.warning(
                f"No records to insert from {npz_file}, skipping page {page_idx}/{total_pages}"
            )
            continue

        # Insert records into Milvus
        try:
            client.insert(collection_name=new_collection_name, data=records)
            logger.info(
                f"Successfully inserted {len(records)} records from page {page_idx}/{total_pages} "
                f"({npz_file}) into collection '{new_collection_name}'"
            )
        except Exception as e:
            logger.error(
                f"Error inserting records from {npz_file} (page {page_idx}/{total_pages}): {e}"
            )
            continue


def create_new_collection(
    client: MilvusClient,
    collection_name: str,
    metadata_config: dict,
    collection_prefix: str | None = None,
) -> None:
    """Create a new Milvus collection with appropriate schema. Drops existing collection if it exists."""
    if client.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists. Dropping it first...")
        client.drop_collection(collection_name)

    # Extract asset type from collection name
    try:
        asset_type = extract_asset_type_from_collection_name(collection_name, collection_prefix)
    except ValueError as e:
        logger.warning(f"{e}, using default schema without metadata")
        asset_type = None

    vector_index_kwargs = {
        "index_type": "HNSW_SQ",
        "metric_type": "COSINE",
        "params": {"sq_type": "SQ8"},
    }
    scalar_index_kwargs = {"index_type": "INVERTED"}

    # Create base schema
    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("asset_id", DataType.VARCHAR, max_length=50)
    schema.add_field("asset_version", DataType.INT64)

    if asset_type in ["datasets", "ml_models", "publications", "educational_resources"]:
        try:
            add_metadata_fields(schema, asset_type, metadata_config)
            logger.info(f"Added metadata fields for asset type '{asset_type}'")
        except Exception as e:
            logger.warning(f"Failed to add metadata fields for asset type '{asset_type}': {e}")

    schema.verify()

    # Create indexes
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", **vector_index_kwargs)
    index_params.add_index(field_name="asset_id", **scalar_index_kwargs)

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    logger.info(f"Created collection '{collection_name}'")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Populate Milvus database with embeddings from .npz/.json files "
        "(output from helper_script.py)."
    )
    parser.add_argument(
        "-i",
        "--input_dirpath",
        type=str,
        required=False,
        default="/data_to_populate",  # Docker compose default value
        help="Path to the directory containing collection folders with .npz (vectors) and .json (asset_ids) files",
    )
    parser.add_argument(
        "--uri",
        type=str,
        required=False,
        help="URI of the local Milvus database",
        default="http://localhost:19530",
    )
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        required=False,
        help="Username of the Milvus user to connect to the local Milvus database",
        default="root",  # Default root username when Milvus DB is brand new
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        required=False,
        help="Password of the Milvus user to connect to the local Milvus database",
        default="Milvus",  # Default root password when Milvus DB is brand new
    )
    parser.add_argument(
        "--metadata-fields-path",
        type=str,
        required=False,
        help="Path to milvus_metadata_fields.json configuration file",
        default="./milvus_metadata_fields.json",  # Docker compose default value
    )
    parser.add_argument(
        "--old-collection-prefix",
        type=str,
        required=False,
        help="Prefix used in existing Milvus collection names/data (e.g., 'AIoD_STS_GTE'). "
        "Used to extract asset type from existing collection names.",
        default="AIoD_STS_GTE",
    )
    parser.add_argument(
        "--new-collection-prefix",
        type=str,
        required=False,
        help="Prefix to use for creating new Milvus collections (e.g., 'aiod'). "
        "New collections will be created with this prefix.",
        default="aiod",
    )

    try:
        args = InputArgs(**parser.parse_args().__dict__)

        start = time()
        populate_database(args)
        end = time()

        logger.info("Database population completed successfully")
        logger.info(f"Population took {int(end - start)} seconds")

        exit(0)
    except Exception as e:
        logger.error(f"Error during database population: {e}", exc_info=True)
        exit(1)
