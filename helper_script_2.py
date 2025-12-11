#!/usr/bin/env python3
"""
Script to find intersection of asset IDs between Milvus exports and MongoDB.

This script:
1. Reads asset IDs from Milvus export files (.json files from helper_script.py output)
2. Reads asset IDs from MongoDB collection (assetsForMetadataExtraction)
3. Finds the intersection of these two sets
4. Optionally removes documents from MongoDB that don't have embeddings
5. Optionally removes Milvus embeddings that don't have MongoDB documents
"""

import json
from pathlib import Path
from typing import Set

import numpy as np
from pymongo import MongoClient
from tqdm import tqdm


def load_milvus_asset_ids(milvus_export_dir: str) -> Set[str]:
    """
    Load all asset IDs from Milvus export all_asset_ids.json file.

    Args:
        milvus_export_dir: Directory containing all_asset_ids.json file

    Returns:
        Set of asset IDs (as strings)
    """
    milvus_path = Path(milvus_export_dir)
    if not milvus_path.exists():
        raise ValueError(f"Milvus export directory does not exist: {milvus_export_dir}")

    all_asset_ids_file = milvus_path / "all_asset_ids.json"
    if not all_asset_ids_file.exists():
        raise ValueError(
            f"all_asset_ids.json not found in {milvus_export_dir}. "
            f"Please run helper_script.py first to generate it."
        )

    print(f"  Reading asset IDs from {all_asset_ids_file.name}...")

    with open(all_asset_ids_file) as f:
        all_collection_asset_ids = json.load(f)

    # Combine all asset IDs from all collections into a single set
    asset_ids = set()
    for collection_name, collection_asset_ids in all_collection_asset_ids.items():
        if isinstance(collection_asset_ids, list):
            asset_ids.update(str(aid) for aid in collection_asset_ids)
        else:
            print(f"    Warning: Unexpected format for collection {collection_name}")

    return asset_ids


def load_mongodb_asset_ids(
    mongo_host: str,
    mongo_port: int,
    mongo_dbname: str,
    mongo_user: str,
    mongo_password: str,
    mongo_auth_dbname: str = "admin",
) -> Set[str]:
    """
    Load all asset IDs from MongoDB collection.

    Args:
        mongo_host: MongoDB host
        mongo_port: MongoDB port
        mongo_dbname: Database name
        mongo_user: MongoDB username
        mongo_password: MongoDB password
        mongo_auth_dbname: Authentication database name

    Returns:
        Set of asset IDs (as strings)
    """
    # Build connection string
    connection_string = (
        f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/"
        f"{mongo_dbname}?authSource={mongo_auth_dbname}"
    )

    client = MongoClient(connection_string)
    db = client[mongo_dbname]
    collection = db["assetsForMetadataExtraction"]

    print("  Querying MongoDB for asset IDs...")

    # Query all documents and extract asset_id field
    asset_ids = set()
    cursor = collection.find({}, {"asset_id": 1})

    for doc in tqdm(cursor, desc="  Loading MongoDB asset IDs"):
        asset_id = doc.get("asset_id")
        if asset_id:
            asset_ids.add(str(asset_id))

    client.close()
    return asset_ids


def remove_mongodb_documents(
    mongo_host: str,
    mongo_port: int,
    mongo_dbname: str,
    mongo_user: str,
    mongo_password: str,
    asset_ids_to_remove: Set[str],
    mongo_auth_dbname: str = "admin",
) -> int:
    """
    Remove documents from MongoDB that don't have corresponding embeddings.

    Args:
        mongo_host: MongoDB host
        mongo_port: MongoDB port
        mongo_dbname: Database name
        mongo_user: MongoDB username
        mongo_password: MongoDB password
        asset_ids_to_remove: Set of asset IDs to remove
        mongo_auth_dbname: Authentication database name

    Returns:
        Number of documents deleted
    """
    if len(asset_ids_to_remove) == 0:
        return 0

    connection_string = (
        f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/"
        f"{mongo_dbname}?authSource={mongo_auth_dbname}"
    )

    client = MongoClient(connection_string)
    db = client[mongo_dbname]
    collection = db["assetsForMetadataExtraction"]

    print(f"  Removing {len(asset_ids_to_remove)} documents from MongoDB...")

    # Delete documents with asset_ids not in the intersection
    result = collection.delete_many({"asset_id": {"$in": list(asset_ids_to_remove)}})
    deleted_count = result.deleted_count

    client.close()
    return deleted_count


def remove_milvus_embeddings(
    milvus_export_dir: str, asset_ids_to_remove: Set[str]
) -> dict[str, int]:
    """
    Remove Milvus embedding files (.npz and .json) that don't have corresponding MongoDB documents.

    This function removes entire pages if all their asset_ids are in the removal set.
    For pages with partial matches, it creates new files with only the kept asset_ids.

    Args:
        milvus_export_dir: Directory containing collection folders
        asset_ids_to_remove: Set of asset IDs to remove

    Returns:
        Dictionary with counts: {"pages_removed": X, "pages_modified": Y, "records_removed": Z}
    """
    milvus_path = Path(milvus_export_dir)
    stats = {"pages_removed": 0, "pages_modified": 0, "records_removed": 0}

    # Iterate through collection folders
    for collection_dir in sorted(milvus_path.iterdir()):
        if not str(collection_dir).endswith("datasets"):
            continue
        if not collection_dir.is_dir():
            continue

        print(f"  Processing {collection_dir.name}...")

        # Find all JSON files matching page_*.json pattern
        json_files = sorted(
            [
                f
                for f in collection_dir.iterdir()
                if f.suffix == ".json" and f.stem.startswith("page_")
            ],
            key=lambda x: int(x.stem.split("_")[1]),
        )

        for json_file in tqdm(json_files, total=len(json_files)):
            npz_file = json_file.with_suffix(".npz")

            if not npz_file.exists():
                print(f"    Warning: {npz_file} not found, skipping {json_file}")
                continue

            try:
                # Load asset IDs and vectors
                with open(json_file) as f:
                    page_asset_ids = json.load(f)

                npz_data = np.load(npz_file)
                vectors = npz_data["vectors"]

                if len(vectors) != len(page_asset_ids):
                    print(f"    Warning: Mismatch in {json_file.name}, skipping")
                    continue

                # Filter out asset_ids to remove
                kept_indices = []
                kept_asset_ids = []
                removed_count = 0

                for idx, asset_id in enumerate(page_asset_ids):
                    if str(asset_id) not in asset_ids_to_remove:
                        kept_indices.append(idx)
                        kept_asset_ids.append(asset_id)
                    else:
                        removed_count += 1

                stats["records_removed"] += removed_count

                # If all records removed, delete the files
                if len(kept_asset_ids) == 0:
                    json_file.unlink()
                    npz_file.unlink()
                    stats["pages_removed"] += 1
                    print(f"    Removed page {json_file.name} (all records removed)")
                # If some records removed, update the files
                elif removed_count > 0:
                    # Save filtered vectors
                    kept_vectors = vectors[kept_indices]
                    np.savez_compressed(npz_file, vectors=kept_vectors)

                    # Save filtered asset IDs
                    with open(json_file, "w") as f:
                        json.dump(kept_asset_ids, f)

                    stats["pages_modified"] += 1
                    print(
                        f"    Modified page {json_file.name} "
                        f"(removed {removed_count}/{len(page_asset_ids)} records)"
                    )

            except Exception as e:
                print(f"    Error processing {json_file.name}: {e}")
                continue

    return stats


def main():
    # ============================================================================
    # CONFIGURATION - Fill in your credentials here
    # ============================================================================

    # Milvus export directory (output from helper_script.py)
    MILVUS_EXPORT_DIR = "./milvus_export"

    # MongoDB connection settings
    MONGO_HOST = "localhost"
    MONGO_PORT = 27018
    MONGO_DBNAME = "aiod"
    MONGO_USER = "aiod"
    MONGO_PASSWORD = "mongopassword"
    MONGO_AUTH_DBNAME = "admin"

    # Cleanup options
    REMOVE_MONGODB = True  # Set to True to remove MongoDB docs without embeddings
    REMOVE_MILVUS = True  # Set to True to remove Milvus embeddings without MongoDB docs
    DRY_RUN = False  # Set to False to actually perform removals

    # ============================================================================
    # MAIN EXECUTION
    # ============================================================================

    print("=" * 80)
    print("Finding intersection of asset IDs between Milvus and MongoDB")
    print("=" * 80)

    # Load asset IDs from Milvus
    print("\n1. Loading asset IDs from Milvus exports...")
    milvus_asset_ids = load_milvus_asset_ids(MILVUS_EXPORT_DIR)
    print(f"   Found {len(milvus_asset_ids)} unique asset IDs in Milvus exports")

    # Load asset IDs from MongoDB
    print("\n2. Loading asset IDs from MongoDB...")
    mongodb_asset_ids = load_mongodb_asset_ids(
        MONGO_HOST,
        MONGO_PORT,
        MONGO_DBNAME,
        MONGO_USER,
        MONGO_PASSWORD,
        MONGO_AUTH_DBNAME,
    )
    print(f"   Found {len(mongodb_asset_ids)} unique asset IDs in MongoDB")

    # Find intersection and differences
    intersection = milvus_asset_ids & mongodb_asset_ids
    only_in_milvus = milvus_asset_ids - mongodb_asset_ids
    only_in_mongodb = mongodb_asset_ids - milvus_asset_ids

    print("\n3. Analysis:")
    print(f"   Intersection (in both): {len(intersection)} asset IDs")
    print(f"   Only in Milvus: {len(only_in_milvus)} asset IDs")
    print(f"   Only in MongoDB: {len(only_in_mongodb)} asset IDs")

    # Save results to JSON
    results = {
        "intersection": sorted(list(intersection)),
        "only_in_milvus": sorted(list(only_in_milvus)),
        "only_in_mongodb": sorted(list(only_in_mongodb)),
        "stats": {
            "milvus_total": len(milvus_asset_ids),
            "mongodb_total": len(mongodb_asset_ids),
            "intersection_count": len(intersection),
            "only_in_milvus_count": len(only_in_milvus),
            "only_in_mongodb_count": len(only_in_mongodb),
        },
    }

    # results_file = Path(MILVUS_EXPORT_DIR) / "intersection_analysis.json"
    # with open(results_file, "w") as f:
    #     json.dump(results, f, indent=2)
    # print(f"\n   Results saved to: {results_file}")

    # Perform cleanup if requested
    if DRY_RUN:
        print("\n" + "=" * 80)
        print("DRY RUN MODE - No changes will be made")
        print("=" * 80)

        if REMOVE_MONGODB:
            print(f"\nWould remove {len(only_in_mongodb)} documents from MongoDB")

        if REMOVE_MILVUS:
            print(f"\nWould remove {len(only_in_milvus)} asset IDs from Milvus exports")
            print("(This would modify/remove page files)")

    else:
        if REMOVE_MONGODB and len(only_in_mongodb) > 0:
            print("\n" + "=" * 80)
            print("Removing MongoDB documents without embeddings...")
            print("=" * 80)
            deleted = remove_mongodb_documents(
                MONGO_HOST,
                MONGO_PORT,
                MONGO_DBNAME,
                MONGO_USER,
                MONGO_PASSWORD,
                only_in_mongodb,
                MONGO_AUTH_DBNAME,
            )
            print(f"   Removed {deleted} documents from MongoDB")

        if REMOVE_MILVUS and len(only_in_milvus) > 0:
            print("\n" + "=" * 80)
            print("Removing Milvus embeddings without MongoDB documents...")
            print("=" * 80)
            stats = remove_milvus_embeddings(MILVUS_EXPORT_DIR, only_in_milvus)
            print(f"   Pages removed: {stats['pages_removed']}")
            print(f"   Pages modified: {stats['pages_modified']}")
            print(f"   Records removed: {stats['records_removed']}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
