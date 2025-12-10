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

if [ "$#" -eq 0 ]; then
  COMPOSE_COMMAND="up -d --build"

  mkdir -p ${DATA_DIRPATH}/volumes
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

docker compose -f docker-compose.mongo.yml -f docker-compose.temp.yml $COMPOSE_COMMAND
