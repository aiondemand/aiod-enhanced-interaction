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

# Check if DEPLOY_FASTAPI_ONLY is set
if [ -z "$DEPLOY_FASTAPI_ONLY" ]; then
  echo "DEPLOY_FASTAPI_ONLY is not set"
  exit 1
fi

# Check if USE_GPU is set
if [ -z "$USE_GPU" ]; then
  echo "USE_GPU is not set"
  exit 1
fi

# Select the base docker-compose file
MAIN_COMPOSE_FILE="docker-compose.yml"
if [ "$DEPLOY_FASTAPI_ONLY" == "true" ]; then
  MAIN_COMPOSE_FILE="docker-compose.standalone.yml"
fi

# What operation we wish to perform
COMPOSE_COMMAND="up -d --build"
if [ "$1" == "--stop" ]; then  
  COMPOSE_COMMAND="stop"
elif [ "$1" == "--remove" ]; then  
  COMPOSE_COMMAND="down"
fi

# Override some properties in order to use GPU for your service
if [ "$USE_GPU" == "true" ]; then
  docker compose -f $MAIN_COMPOSE_FILE -f docker-compose.gpu.yml $COMPOSE_COMMAND
else
  docker compose -f $MAIN_COMPOSE_FILE $COMPOSE_COMMAND
fi