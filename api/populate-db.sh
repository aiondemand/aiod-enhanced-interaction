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

# Check if INITIAL_EMBEDDINGS_TO_POPULATE_DB_WITH_DIRPATH is set
if [ -z "$INITIAL_TINYDB_JSON_FILEPATH" ]; then
  echo "INITIAL_TINYDB_JSON_FILEPATH is not set"
  exit 1
fi

# Check if DATA_DIRPATH is set
if [ -z "$DATA_DIRPATH" ]; then
  echo "DATA_DIRPATH is not set"
  exit 1
fi

echo "lets go"
docker compose -f docker-compose.yml -f docker-compose.populate.yml up populate-db --build
EXIT_CODE=$?
docker compose -f docker-compose.yml -f docker-compose.populate.yml down

if [ $EXIT_CODE -eq 0 ]; then
  echo "Population of vector DB has run successfully."
  mkdir -p ${DATA_DIRPATH}/volumes/tinydb/
  cp $INITIAL_TINYDB_JSON_FILEPATH ${DATA_DIRPATH}/volumes/tinydb/tinydb.json
  EXIT_CODE_2=$?

  if [ $EXIT_CODE_2 -eq 0 ]; then
    echo "Moved tinydb.json file onto its proper position"
    echo "=== SETUP COMPLETE SUCCESSFULLY ==="
  else
    echo "Failed to moved tinydb.json"
  fi
  
else
  echo "Population of vector DB has encountered some ERRORS."
fi

