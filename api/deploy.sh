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

# Select the base docker-compose file
MAIN_COMPOSE_FILE="docker-compose.yml"

# What operation we wish to perform
COMPOSE_COMMAND="up -d --build"
if [ "$1" == "--stop" ]; then
  COMPOSE_COMMAND="stop"
elif [ "$1" == "--remove" ]; then
  COMPOSE_COMMAND="down"
fi

# TODO we need to create docker compose dynamically or through some templates as this is not scalable...
# Note: This has been already implemented in other branch
if [ "$USE_GPU" = "true" ] && [ "$USE_LLM" = "true" ]; then
  docker compose -f $MAIN_COMPOSE_FILE -f docker-compose.ollama.yml -f docker-compose.gpu.yml -f docker-compose.ollama.gpu.yml $COMPOSE_COMMAND
elif [ "$USE_GPU" = "true" ]; then
  docker compose -f $MAIN_COMPOSE_FILE -f docker-compose.gpu.yml $COMPOSE_COMMAND
elif [ "$USE_LLM" = "true" ]; then
  docker compose -f $MAIN_COMPOSE_FILE -f docker-compose.ollama.yml $COMPOSE_COMMAND
else
  docker compose -f $MAIN_COMPOSE_FILE $COMPOSE_COMMAND
fi
