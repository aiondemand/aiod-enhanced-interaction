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

# Check if USE_GPU is set
if [ -z "$USE_GPU" ]; then
  echo "USE_GPU is not set"
  exit 1
fi

# Check if USE_LLM is set
if [ -z "$USE_LLM" ]; then
  echo "USE_LLM is not set"
  exit 1
fi

# Check if DATA_DIRPATH is set
if [ -z "$DATA_DIRPATH" ]; then
  echo "DATA_DIRPATH is not set"
  exit 1
fi

# What operation we wish to perform
if [ "$#" -eq 0 ]; then
  COMPOSE_COMMAND="up -d --build"

  # Build docker-compose first (stored as docker-compose.final.yml)
  docker compose -f docker-compose.build.yml up --build
  CONTAINER_NAME="${COMPOSE_PROJECT_NAME}-build-compose-1"
  EXIT_CODE=$(docker inspect $CONTAINER_NAME --format='{{.State.ExitCode}}')
  docker compose -f docker-compose.build.yml down

  # Create folders for holding Docker volumes if they don't exist
  # otherwise Milvus would create them under the root user...
  mkdir -p ${DATA_DIRPATH}/volumes
  mkdir -p ${DATA_DIRPATH}/model
  mkdir -p ${DATA_DIRPATH}/cold_data

  if [ $EXIT_CODE -ne 0 ]; then
    echo "Failed to build a docker compose"
    exit 1
  fi
elif [ "$#" -eq 1 ]; then
  if [ "$1" == "--stop" ]; then
    COMPOSE_COMMAND="stop"
  elif [ "$1" == "--remove" ]; then
    COMPOSE_COMMAND="down"
  else
    echo "Invalid operation '$1'. Only '--stop' and '--remove' are allowed"
    exit 1
  fi
else
  echo "Invalid number of arguments"
  exit 1
fi

docker compose -f docker-compose.milvus.yml -f docker-compose.mongo.yml -f docker-compose.final.yml $COMPOSE_COMMAND
