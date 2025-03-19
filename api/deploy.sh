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

# What operation we wish to perform
if [ "$#" -eq 0 ]; then
  COMPOSE_COMMAND="up -d --build"

  # Build docker-compose first (stored as docker-compose.final.yml)
  docker compose -f docker-compose.build.yml up --build
  EXIT_CODE=$?
  docker compose -f docker-compose.build.yml down

  if [ $EXIT_CODE -ne 0 ]; then
    echo "Failed to build a docker compose"
    exit 1
  fi
elif [ "$1" == "--stop" ]; then
  COMPOSE_COMMAND="stop"
elif [ "$1" == "--remove" ]; then
  COMPOSE_COMMAND="down"
else
  echo "Invalid operation '$1'. Only '--stop' and '--remove' are allowed"
  exit 1
fi

# TODO uncomment
# docker compose -f docker-compose.milvus.yml -f docker-compose.final.yml $COMPOSE_COMMAND
