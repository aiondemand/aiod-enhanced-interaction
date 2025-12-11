#!/usr/bin/env python3
"""
Script to migrate Milvus collection data to files.

For each collection:
- Creates a folder named after the collection
- Exports documents in pages (default: 1000 documents per page)
- Each page has two files:
  * .npz file: Contains vectors as a 2D numpy array (num_docs x vector_dim)
  * .json file: Contains asset_ids as a list of strings
- Collects all asset IDs and saves them in a separate JSON file
"""

import json
from pathlib import Path
import numpy as np

from pymilvus import MilvusClient
from tqdm import tqdm

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


def get_collection_name(prefix: str, asset_name: str) -> str:
    """
    Create a Milvus collection name from a prefix and asset name.

    Args:
        prefix: Collection prefix (e.g., "aiod")
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


def connect_to_milvus(uri: str, user: str, password: str) -> MilvusClient:
    """Connect to Milvus database."""
    token = f"{user}:{password}"
    client = MilvusClient(uri=uri, token=token)
    return client


def migrate_collection(
    client: MilvusClient,
    collection_name: str,
    output_base_dir: Path,
    page_size: int = 1000,
    testing: bool = False,
) -> set[str]:
    """Migrate a single collection to JSON files."""
    print(f"\nProcessing collection: {collection_name}")

    # Create folder for this collection
    collection_dir = output_base_dir / collection_name
    collection_dir.mkdir(parents=True, exist_ok=True)

    # Check if collection exists
    if not client.has_collection(collection_name):
        print(f"  Warning: Collection '{collection_name}' does not exist, skipping...")
        return set()

    # Load collection
    try:
        client.load_collection(collection_name)
    except Exception as e:
        print(f"  Error loading collection: {e}")
        return set()

    # Determine which fields to query
    # Include "id" field to track pagination
    output_fields = ["id", "vector", "asset_id"]

    # Process documents in pages using ID-based pagination
    # This circumvents the limit + offset < 16384 constraint
    all_asset_ids = set()
    page_num = 0
    processed = 0
    last_id = 0  # Start with id > 0
    max_pages = 3 if testing else None

    if testing:
        print(f"  TESTING MODE: Processing only first {max_pages} pages")

    print(f"  Querying documents page by page using ID-based pagination...")

    while True:
        # Stop after max_pages in testing mode
        if testing and page_num >= max_pages:
            print(f"  TESTING MODE: Stopping after {max_pages} pages")
            break

        # Build filter: id > last_id (or id > 0 for first page)
        filter_expr = f"id > {last_id}"

        # Query one page of documents
        try:
            page_docs = list(
                client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=output_fields,
                    limit=page_size,
                )
            )
        except Exception as e:
            print(f"  Error querying collection with filter '{filter_expr}': {e}")
            break

        # If no more documents, we're done
        if len(page_docs) == 0:
            print(f"  No more documents found (reached end of collection)")
            break

        max_id_in_page = page_docs[-1]["id"]

        # Process the page: collect vectors and asset_ids separately
        page_vectors = []
        page_asset_ids = []

        for doc in page_docs:
            try:
                # Extract both vector and asset_id first
                vector = doc.get("vector", [])
                asset_id = doc.get("asset_id", "")

                # Skip documents without asset_id
                if not asset_id:
                    print(f"  Warning: Document has no asset_id, skipping")
                    continue

                # Skip documents without vector
                if not vector:
                    print(f"  Warning: Document has no vector, skipping")
                    continue

                # Convert vector to numpy array
                if isinstance(vector, (list, np.ndarray)):
                    vector_array = np.hstack(vector)
                    # Flatten if needed (handle nested lists)
                    if vector_array.ndim > 1:
                        vector_array = vector_array.flatten()
                    page_vectors.append(vector_array)
                else:
                    print(f"  Warning: Unexpected vector format, skipping document")
                    continue

                # Add asset_id (both to page list and all_asset_ids set)
                page_asset_ids.append(str(asset_id))
                all_asset_ids.add(str(asset_id))

            except Exception as e:
                print(f"  Warning: Error processing document: {e}")
                continue

        # Save page data: vectors as .npz and asset_ids as .json
        if page_vectors and page_asset_ids:
            # Ensure vectors and asset_ids have the same length (they should, but double-check)
            if len(page_vectors) != len(page_asset_ids):
                print(
                    f"  Warning: Mismatch between vectors ({len(page_vectors)}) and asset_ids ({len(page_asset_ids)}), truncating"
                )
                min_len = min(len(page_vectors), len(page_asset_ids))
                page_vectors = page_vectors[:min_len]
                page_asset_ids = page_asset_ids[:min_len]

            # Stack vectors into 2D array (num_docs x vector_dim)
            vectors_array = np.stack(page_vectors)

            # Save vectors as .npz file
            npz_filename = collection_dir / f"page_{page_num:04d}.npz"
            np.savez_compressed(npz_filename, vectors=vectors_array)

            # Save asset_ids as JSON file (just a list of strings)
            json_filename = collection_dir / f"page_{page_num:04d}.json"
            with open(json_filename, "w") as f:
                json.dump(page_asset_ids, f)

        processed += len(page_asset_ids)
        page_num += 1

        print(f"  Progress: {processed} documents ({page_num} pages, last_id: {max_id_in_page})")

        # If we got fewer documents than page_size, we've reached the end
        if len(page_docs) < page_size:
            break

        # Update last_id for next page (use max_id_in_page as the threshold)
        last_id = max_id_in_page

    print(f"  Completed: {processed} documents in {page_num} pages")
    print(f"  Unique asset IDs: {len(all_asset_ids)}")

    return all_asset_ids


def main():
    """Main migration function."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate Milvus collections to JSON files")
    parser.add_argument(
        "--uri",
        type=str,
        default="http://localhost:19530",
        help="Milvus URI (default: from MILVUS_URI env var or http://localhost:19530)",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="aiod",
        help="Milvus username (default: from MILVUS_USER env var or 'root')",
    )
    parser.add_argument(
        "--password",
        type=str,
        default="XXX",
        help="Milvus password (default: from MILVUS_PASS env var or 'Milvus')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./milvus_export",
        help="Output directory for exported JSON files (default: ./milvus_export)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=10_000,
        help="Number of documents per JSON file (default: 1000)",
    )
    parser.add_argument(
        "--collections",
        type=str,
        nargs="*",
        default=None,
        help="Specific collections to migrate (default: all collections)",
    )
    parser.add_argument(
        "--testing",
        default=False,
        help="Testing mode: process only 3 pages from each collection",
    )

    args = parser.parse_args()

    # Connect to Milvus
    print("Connecting to Milvus...")
    try:
        client = connect_to_milvus(args.uri, args.user, args.password)
        print("Connected successfully!")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return 1

    collection_names = [get_collection_name("AIoD_STS_GTE", asset) for asset in VALID_ASSET_TYPES]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")

    # Migrate each collection
    all_collection_asset_ids = {}

    for collection_name in tqdm(collection_names, desc="Collections"):
        try:
            asset_ids = migrate_collection(
                client, collection_name, output_dir, args.page_size, args.testing
            )
            all_collection_asset_ids[collection_name] = list(asset_ids)
        except Exception as e:
            print(f"\nError migrating collection {collection_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save all asset IDs to a separate JSON file
    asset_ids_file = output_dir / "all_asset_ids.json"
    with open(asset_ids_file, "w") as f:
        json.dump(all_collection_asset_ids, f, indent=2)

    print(f"\n✓ Migration complete!")
    print(f"  Exported {len(collection_names)} collections")
    print(f"  Asset IDs saved to: {asset_ids_file}")
    print(f"  Output directory: {output_dir.absolute()}")

    return 0


if __name__ == "__main__":
    main()
