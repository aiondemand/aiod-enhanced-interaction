#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo ".env file not found"
  exit 1
fi

# Check if INITIAL_EMBEDDINGS_TO_POPULATE_DB_WITH_DIRPATH is set
if [ -z "$INITIAL_EMBEDDINGS_TO_POPULATE_DB_WITH_DIRPATH" ]; then
  echo "INITIAL_EMBEDDINGS_TO_POPULATE_DB_WITH_DIRPATH is not set"
  exit 1
fi

# Check if MONGO_DUMP_DIRPATH is set
if [ -z "$MONGO_DUMP_DIRPATH" ]; then
  echo "$MONGO_DUMP_DIRPATH is not set"
  exit 1
fi

# Check if DATA_DIRPATH is set
if [ -z "$DATA_DIRPATH" ]; then
  echo "DATA_DIRPATH is not set"
  exit 1
fi

# Check if USE_MILVUS_LITE is set
if [ -z "$USE_MILVUS_LITE" ]; then
  echo "USE_MILVUS_LITE is not set"
  exit 1
fi

# Create path under the current user so that it won't be created automatically by Milvus under root
mkdir -p ${DATA_DIRPATH}/volumes

# Run Milvus and Mongo databases
docker compose -f docker-compose.mongo.yml up -d

if [ "$USE_MILVUS_LITE" = "true" ]; then
  docker compose -f docker-compose.populate_lite.yml up populate-db --build
else
  docker compose -f docker-compose.milvus.yml -f docker-compose.populate.yml up populate-db --build
fi

CONTAINER_NAME="${COMPOSE_PROJECT_NAME}-populate-db-1"
EXIT_CODE=$(docker inspect $CONTAINER_NAME --format='{{.State.ExitCode}}')

if [ $EXIT_CODE -eq 0 ]; then
  echo "Population of vector DB has run successfully."

  CONTAINER_NAME_MONGO="${COMPOSE_PROJECT_NAME}-mongo-1"

  docker cp "${MONGO_DUMP_DIRPATH}/assetCollections.json" ${CONTAINER_NAME_MONGO}:.
  docker exec $CONTAINER_NAME_MONGO mongoimport --db=aiod --collection=assetCollections --file=assetCollections.json --jsonArray --username=${MONGO_USER} --password=${MONGO_PASSWORD} --authenticationDatabase=${MONGO_AUTH_DBNAME:=admin}
  EXIT_CODE_2a=$?

  if [ "$USE_MILVUS_LITE" = "false" ]; then
    # We need to import assets for metadata extraction only in the case we don't use Milvus Lite
    docker cp "${MONGO_DUMP_DIRPATH}/assetsForMetadataExtraction.json" ${CONTAINER_NAME_MONGO}:.
    docker exec $CONTAINER_NAME_MONGO mongoimport --db=aiod --collection=assetsForMetadataExtraction --file=assetsForMetadataExtraction.json --jsonArray --username=${MONGO_USER} --password=${MONGO_PASSWORD} --authenticationDatabase=${MONGO_AUTH_DBNAME:=admin}
  fi
  EXIT_CODE_2b=$?

  if [ $EXIT_CODE_2a -eq 0 ] && [ $EXIT_CODE_2b -eq 0 ]; then
    echo "Migration of MongoDB database has run successfully"
    echo "=== SETUP COMPLETE SUCCESSFULLY ==="
  else
    echo "Failed to populate MongoDB"
  fi
else
  echo "Population of vector DB has encountered some ERRORS."
fi

# Shutdown databases
if [ "$USE_MILVUS_LITE" = "true" ]; then
  docker compose -f docker-compose.mongo.yml -f docker-compose.populate_lite.yml down
else
  docker compose -f docker-compose.milvus.yml -f docker-compose.mongo.yml -f docker-compose.populate.yml down
fi
