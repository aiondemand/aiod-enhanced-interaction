# Semantic search module

<!-- TODO UPDATE (account for other services besides sem search when describing this repo) -->

<!-- TODO Describe the chatbot and what tools it has an access to -->

This API service expands the AIoD platform by providing a separately managed
semantic search module that can be applied to AIoD assets,
i.e, Datasets, Models, Publications, etc.

This API service contains the following key functionalities:
- API endpoints for retrieving the most similar assets to a user query
- Retrieving and storing asset embeddings into a embedding store (Milvus)

As of right now, this service utilizes a [GTE large model](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) to embed the assets into
multiple chunks depending on the lengths of the input documents. Each supported asset
type is stored in its own vector collection.

Currently supported AIoD asset types:
- Dataset
- ML_Model
- Publication
- Case studies
- Educational resources
- Experiments
- Services

## Acquisition of AIoD assets

In order to keep track of all the AIoD asset changes and subsequently propagate them into embedding store, we
prompt the AIoD API for detected asset changed in regular intervals.

The initial population of the embedding store is performed immediately after the application is executed for the first time.
In this case, we iterate over all the AIoD assets in order to embed them. Once the database is filled, we perform the aforementioned recurring job of
detecting changes made to AIoD assets.

## User query handling

After you invoke API endpoint in hopes of retrieving most relevant assets to your query, said query is put onto a queue to be processed and you're given a
URL you can expect the results to be at.

To maintain the stability of our application, so far we only support handling of one user query at once, hence the use of a query queue. After the specific query
is processed, the results are saved in a local nosql file-based database and are accessible to the end users via the URL they were given.

## Metadata filtering

Metadata filtering functionality consists of two main processes that need to be taken care of:
a) Extracting and storing of metadata associated with individual assets
b) Extracting of filters/conditions found in the user query /user query parsing/

### Extracting of metadata from assets

Currently, this process applicability is restricted to only a specific subset of assets we can <ins>manually extract metadata</ins> from, from <ins>Huggingface datasets</ins> to be exact. Due to this extraction process being very time demanding, we have opted to perform the metadata extraction manually without any use of LLMs for now.

Since the datasets as an asset type is the most prevalent asset in the AIoD platform, we have decided to apply metadata filtering on said asset, whilst choosing solely Huggingface datasets that share a common metadata structure as it can be used to retrieve a handful of metadata fields to be stored in Milvus database.

### Extracting of filters from user queries

Since we wish to minimize the costs and increase computational/time efficiency pertaining to serving of an LLM, we have opted to go for a rather small LLM (Llama 3.1 8B). Due to its size, performing more advanced tasks or multiple tasks at once can often lead incorrect results. To mitigate this to a certain degree, we have divided the <ins>user query parsing process into 2-3 LLM steps</ins> that further dissect and scrutinize a user query on different levels of granularity. The LLM steps are:
- STEP 1: Extraction of natural language conditions from query (extraction of spans representing a conditions/filters each associated with a specific metadata field)
- STEP 2: Analysis and transformation of each natural language condition (a span from user query) to a structure representing the condition performed separately
- [Optional STEP 3]: Further validation of transformed value against a list of permitted values for a particular metadata field

Unlike the former process, the extraction of metadata from assets, that can be performed manually without the use of an LLM to only a limited degree, <ins>the manual approach of defining filters explicitly</ins> is a full-fledged alternative to an automated user query parsing by an LLM. To this end, a user can define the filters himself, which can eliminate possibility for an LLM to misinterpret or omit some conditions user wanted to apply in the first place.

---

# Repo Setup

In this section, we describe the necessary steps to take to set up this repository for either development or deployment purposes.

## Environment variables and configs
*When specifying a path to a file or a directory, either specify the fullpath or prepend the relative path with `./` so that it is clear it's an actual path and it will be resolved as such.*

Regardless whether you want to further develop this codespace or deploy the service, you need to create `.env.app` file that can be created from the `.env.app.sample` template.
In this file you find the following ENV variables:
- `API_VERSION`: Prefix string of the latest version of endpoints
- `USE_GPU`: Boolean value that denotes whether you wish to use a GPU for the initial population of Milvus database or not. *(Is overwritten in docker-compose.yml)*
- `MODEL_LOADPATH`: String representing the name of the model to either download from HuggingFace or load from the files locally found on your machine. The string must either be:
    - `Alibaba-NLP/gte-large-en-v1.5`: To download the specific model from the HuggingFace
    - `<PATH_TO_LOCAL_MODEL>`: To load the weights of GTE large model from the local path on your machine / in the container
- `MODEL_BATCH_SIZE`: Number of assets model can compute embeddings for in parallel
- `CONNECTION_NUM_RETRIES`: Number of retry attempts when connecting to external services
- `CONNECTION_SLEEP_TIME`: Sleep time in seconds between retry attempts when connecting to services
- `QUERY_EXPIRATION_TIME_IN_MINUTES`: Time in minutes after which query results expire and are removed
- `LOGFIRE_TOKEN`: API Token for Logfire service that monitors LLM traces
- Milvus config (`MILVUS__*`):
    - `MILVUS__FILEPATH`: A path to the a local file representing the contents of the database. If you wish not to deploy a fully-fledged Milvus database, but rather want something light-weight but limited in functionality, you can opt for Milvus Lite.
        - **Important:** If you specify this variable, you automatically opt for using Milvus Lite and you can't perform the Metadata Filtering functionality
        - *We suggest using Milvus Lite only when the data you wish to vectorize is small enough (less than 1m vectors)*
    - `MILVUS__URI`: URI of the Milvus database server. *(Is overwritten in docker-compose.yml)*
    - `MILVUS__USER`: Username of the user to log into the Milvus database *(Is overwritten in docker-compose.yml)*
    - `MILVUS__PASS`: Password of the user to log into the Milvus database *(Is overwritten in docker-compose.yml)*
    - `MILVUS__COLLECTION_PREFIX`: Prefix to use for naming our Milvus collections
    - `MILVUS__BATCH_SIZE`: Number of embeddings to accumulate into batch before storing it in Milvus database
    - `MILVUS__STORE_CHUNKS`: Boolean value that denotes whether we wish to store the embeddings of the individual chunks of each document or to have only one embedding representing the entire asset.
- Mongo config (`MONGO__*`):
    - `MONGO__HOST`: Hostname of MongoDB database. *(Is overwritten in docker-compose.yml)*
    - `MONGO__PORT`: Port of the database within MongoDB to store our data in *(Is overwritten in docker-compose.yml)*
    - `MONGO__USER`: Username of the admin of the database within MongoDB to store our data in
    - `MONGO__PASSWORD`: Password of the admin of the database within MongoDB to store our data in
    - `MONGO__DBNAME`: Name of the database within MongoDB to store our data in
    - `MONGO__AUTH_DBNAME`: Name of the database within MongoDB we use for authentication

- Metadata filtering config (`METADATA_FILTERING__*`):
    - `METADATA_FILTERING__ENABLED`: Boolean value that enables/disables metadata filtering functionality
    - `METADATA_FILTERING__ENFORCE_ENUMS`: Boolean value that enforces enum validation for metadata field values (this leads to performing additional checks and multiple LLM passes over the same assets)
- Ollama environment variables (You can omit these if you don't plan on using LLM for metadata filtering (`METADATA_FILTERING__ENABLED` is set to False))
    - `OLLAMA__URI`: URI of the Ollama server.
    - `OLLAMA__MODEL_NAME`: Name of an Ollama model we wish to use for metadata filtering purposes.
    - `OLLAMA__MAX_TOKENS`: The maximum number of tokens an LLM generates.
- Crawler settings (`CRAWLER__*`):
    - `CRAWLER__COMMA_SEPARATED_WEBSITES`: Comma-separated list of website URLs to crawl for chatbot knowledge base
    - `CRAWLER__COMMA_SEPARATED_API_WEBSITES`: Comma-separated list of API website URLs to crawl for chatbot knowledge base
    - `CRAWLER__COMMA_SEPARATED_BLOCKED_WEBSITES`: Comma-separated list of website domains to block during crawling
- Chatbot config (`CHATBOT__*`):
    - `CHATBOT__USE_CHATBOT`: Boolean value that denotes whether you wish to enable the chatbot functionality
    - `CHATBOT__MISTRAL_KEY`: API key required to access the Mistral AI API for chatbot functionality
    - `CHATBOT__MISTRAL_MODEL`: Name of the Mistral AI model to use for chatbot responses
    - `CHATBOT__TOP_K_ASSETS_TO_SEARCH`: Number of top assets to retrieve when searching for assets in chatbot responses
    - `CHATBOT__MYLIBRARY_URL`: Base URL for the MyLibrary service used to generate links to AIoD assets
- AIoD config (`AIOD__*`):
    - `AIOD__URL`: URL of the AIoD API we use to retrieve information about the assets and assets themselves.
    - `AIOD__COMMA_SEPARATED_ASSET_TYPES`: Comma-separated list of values representing all the asset types we wish to process
    - `AIOD__COMMA_SEPARATED_ASSET_TYPES_FOR_METADATA_EXTRACTION`: Comma-separated list of values representing all the asset types we wish to apply metadata filtering on. Only include an asset type into this list if all the setup regarding metadata filtering (manual/automatic extraction of metadata from assets, automatic extraction of filter in user queries)
    - `AIOD__WINDOW_SIZE`: Asset window size (limit of pagination) we use for retrieving assets from AIoD API during the initial setup, by iterating over all the AIoD assets.
    - `AIOD__WINDOW_OVERLAP`: Asset window overlap representing relative size of an overlap we maintain between the pages in pagination. The overlap is necessary so that we wouldn't potentionally skip on some new assets to process if any particular assets were to be deleted in parallel with our update logic, making the whole data returned by AIoD platform slightly shifted.
    - `AIOD__JOB_WAIT_INBETWEEN_REQUESTS_SEC`: Number of seconds we wait when performing JOBs (for updating/deleting assets) in between AIoD requests in order not to overload their API.
    - `AIOD__SEARCH_WAIT_INBETWEEN_REQUESTS_SEC`: Number of seconds we wait in between AIoD requests in order not to overload their API. This wait is used when validating the existence of retrieved Milvus assets that are supposed to be the most relevant to a user query.
    - `AIOD__DAY_IN_MONTH_FOR_EMB_CLEANING`: The day of the month we wish to perform Milvus embedding cleaning. We compare the stored documents to the AIoD platform and delete the embeddings corresponding to old assets that are no longer present on AIoD.
    - `AIOD__DAY_IN_MONTH_FOR_TRAVERSING_ALL_AIOD_ASSETS`: The day of the month we wish to perform recurrent AIoD update that iterates over all the data on said platform rather than only examining assets updated within a specific timeframe. The objective of this particular database update is to double-check we have not missed any new AIoD assets that might've been overlooked due to large number of assets having been deleted in the past.
    - `AIOD__TESTING`: Boolean value you should keep set to false unless you intentionally wish to retrieve only a fraction of all the AIoD assets. This variable is used for testing purposes only
    - `AIOD__STORE_DATA_IN_JSON`: Boolean value you should keep set to false unless you wish to store AIoD embeddings and metadata in JSON files for subsequent database population process on another machine. This flag is especially useful if you don't have an access to GPU or LLM for embedding computation / metadata extraction on the production machine for instance, but you do possess one on your local setup. This way you can precompute all the information to be stored into the Milvus DB and simply migrate said data onto production to expedite the deployment process.
    - `AIOD__JSON_SAVEPATH`: Path where to store JSON files if `AIOD__STORE_DATA_IN_JSON` env var is set to True

## Development

For the development purposes we recommend installing the necessary Python packages locally and develop/debug the codespace on the go.
We don't have any development environment set up utilizing Docker containers.

### Python Environment

Create a Python v11 environment preferably using conda:
- `conda create --name aiod-env python=3.11`
- `conda activate aiod-env; pip install .`
- Optional: `playwright install` (Required when `CHATBOT__USE_CHATBOT` is set to True)

### Dependencies

For the application to work you need to set up its dependencies, namely Milvus database and Ollama. Unless you have an access to already existing services to these technologies, you need to run them locally on your machine for development purposes.
If you decide to run these dependencies using our docker-compose files (further described below), you are also required to create `.env` file with a subset of environment variables that are necessary for these deployments to be executed.
The `.env` file and all its environment variables are described in great detail in the *Deployment* section below.

**MongoDB**
- We recommend deploying the database using `docker-compose.mongo.yml` file
- Required env vars to define in `.env`:
    - Where to store data: `DATA_DIRPATH`
    - Admin username: `MONGO_USER`
    - Admin password: `MONGO_PASSWORD`
- Optional env vars to define in `.env`:
    - Localhost port: `MONGO_HOST_PORT`
- Command to run: `docker compose -f docker-compose.mongo.yml up -d`
- Set the following env vars defined in `.env.app`:
    - `MONGO__HOST=localhost`
    - `MONGO__PORT=<PLACEHOLDER>`
        - Replace the placeholder with the localhost port you have assigned to the Mongo container's 27017 port (specified in the `MONGO_HOST_PORT` env var)
    - `MONGO__USER=<PLACEHOLDER>`
        - Replace the placeholder with the value in `MONGO_USER` env var
    - `MONGO__PASSWORD=<PLACEHOLDER>`
        - Replace the placeholder with the value in `MONGO_PASSWORD` env var

**Milvus**
- We recommend deploying the database using `docker-compose.milvus.yml` file
- Required env vars to define in `.env`:
    - Where to store data: `DATA_DIRPATH`
- Optional env vars to define in `.env`:
    - Minio credentials: `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
    - Minio ports: `MINIO_HOST_PORT_9001`, `MINIO_HOST_PORT_9000`
    - Milvus ports: `MILVUS_HOST_PORT_19530`, `MILVUS_HOST_PORT_9091`
- Command to run: `docker compose -f docker-compose.milvus.yml up -d`
- Set the following env vars defined in `.env.app`:
    - `MILVUS__URI=http://localhost:<PLACEHOLDER>`
        - Replace the placeholder with the localhost port you have assigned to the Milvus container's 19530 port (specified in the `MILVUS_HOST_PORT_19530` env var)
    - `MILVUS__USER=root`
    - `MILVUS__PASS=Milvus`

**Ollama**
- We recommend installing Ollama directly from their website: https://ollama.com/ *(So that you don't need to bother with granting Docker an access to your GPU)*

### Run the application

To start the application you can either:
- Use a .vscode launch option called `Debug FastAPI service` if you use VSCode as your IDE
- Execute the following command: `uvicorn app.main:app --reload`

## Deployment

For deploying purposes, we use a Docker compose config enabling us to deploy not only FastAPI service, but also Milvus vector database, and Ollama service if need be.

*When specifying a path to a file or a directory, either specify the fullpath or prepend the relative path with `./` so that it is clear it's an actual path and it will be resolved as such.*

Perform the following steps to deploy the service:
1. Create additional `.env` file (from `.env.sample` template) containing additional ENV variables to further modify the deployment. To be specific, said file contains the following ENV variables:
    - `COMPOSE_PROJECT_NAME`: A project prefix for all the Docker containers that are deployed within this repository
    - `USER_UID`: User ID of the local non-root user we use for deploying the application. We need this ID to create a non-root user (without sudo privileges) within the Docker image that shares the same IDs for maintaining proper bind mount volumes. Execute the following command in the terminal to retrieve your user ID: `id -u`
    - `USER_GID`: Group ID of the local non-root user we use for deploying the application. We need this ID to create a non-root user (without sudo privileges) within the Docker image that shares the same IDs for maintaining proper bind mount volumes.
    Execute the following command in the terminal to retrieve your group ID: `id -g`
    - `DATA_DIRPATH`: Path to a directory that should contain all the volumes and other files related to our the services we wish to deploy.
    - `USE_GPU`: Boolean value that denotes whether you wish to use a GPU for the initial population of Milvus database or not.  *(Overrides value set by `USE_GPU` in `env.app`)
    - `USE_LLM`: Whether we wish to locally deploy an Ollama service for serving an LLM that can be utilized for metadata extraction and processing. If set to False, we won't support these more advanced asset search processes.
    - `USE_CHATBOT`: Whether we wish to expose endpoints that allow users to communicate with AIoD chatbot. By enabling this variable, we also automatically crawl AIoD platform on daily basis to retrieve the most up-to-date information the chatbot can subsequently have an access to.
    - `INITIAL_EMBEDDINGS_TO_POPULATE_DB_WITH_DIRPATH`: An optional variable representing a dirpath to a specific directory containing a list of JSONs representing precomputed embeddings for various assets. This variable is useful for migrating embeddings on machines that do not possess a GPU unit to increase the computational speed associated with the embedding computations. This variable is specifically tailored for original developers of this repo to expedite the deployment process on AIoD platform.
    - `MONGO_DUMP_DIRPATH`: An optional variable representing a dirpath to a directory containing JSON files representing a list of documents used for populating MongoDB database, more specifically the `assetCollections` and `assetsForMetadataExtraction` collections. This variable is useful for migrating embeddings on machines that do not possess a GPU unit to increase the computational speed associated with the embedding computations. This variable is specifically tailored for original developers of this repo to expedite the deployment process on AIoD platform.

    - Milvus credentials to use/initiate services with:
        - `MILVUS_NEW_ROOT_PASS`: New root password used to replace a default one. The password change is only performed during the first initialization of the Milvus service.
        - `MILVUS_AIOD_USER`: Username of the user to log into the Milvus database. During the Milvus initialization, a user with these credentials is created. *(Overrides value set by `MILVUS__USER` in `env.app`)*
        - `MILVUS_AIOD_PASS`: Password of the user to log into the Milvus database. During the Milvus initialization, a user with these credentials is created. *(Overrides value set by `MILVUS__PASS` in `env.app`)*

    - Minio credential setup:
        - `MINIO_ACCESS_KEY`: Access key to connect to Minio service (During the Milvus initialization, these credentials are used to set up the authorization)
        - `MINIO_SECRET_KEY`: Secret key to connect to Minio service (During the Milvus initialization, these credentials are used to set up the authorization)

    - Mongo credential setup:
        - `MONGO_USER`: Username of the admin MongoDB user
        - `MONGO_PASSWORD`: Password of the admin MongoDB user
        - `MONGO_AUTH_DBNAME`: Name of the auth DB (only used in `./scripts/populate-db.sh`)

    - Mapping of host ports to Docker services:
        - `APP_HOST_PORT`: Host port we wish the FastAPI service to be mapped to
        - `MINIO_HOST_PORT_9001`: Host port we wish the Minio service to be mapped to (Minio container port: 9001)
        - `MINIO_HOST_PORT_9000`: Host port we wish the Minio service to be mapped to (Minio container port: 9000)
        - `MILVUS_HOST_PORT_19530`: Host port we wish the Milvus service to be mapped to (Milvus container port: 19530)
        - `MILVUS_HOST_PORT_9091`: Host port we wish the Milvus service to be mapped to (Milvus container port: 9091)
        - `MONGO_HOST_PORT`: Host port we wish the MongoDB service to be mapped to (MongoDB container port: 27017)
        - `OLLAMA_HOST_PORT`: Host port we wish the Ollama service to be mapped to (Ollama container port: 11434)

1. [Optional] If you wish to download the model weights locally, perform the following steps. Otherwise, to download the model from HuggingFace during runtime, simply keep the `MODEL_LOADPATH` variable set to `Alibaba-NLP/gte-large-en-v1.5`.
    1. Download the model weights and place them into the following directory: `$DATA_DIRPATH/model`. This directory is a Docker mount-bind mapped into the FastAPI container, specifically onto the `/model` path in the container.
    1. Set the `MODEL_LOADPATH` variable accordingly, so that it points to the model weights. This ENV variable needs to point inside the `/model` directory where the model weights are accessible to the Docker container.
1. If you wish to populate the vector database and MongoDB with already precomputed embeddings or other data respectively, set the `INITIAL_EMBEDDINGS_TO_POPULATE_DB_WITH_DIRPATH` and `MONGO_DUMP_DIRPATH` variables and then execute the following bash script that takes care of populating the database: `./scripts/populate-db.sh`. This script is blocking, so you can have a direct feedback whether it finishes successfully or not. It will print out its status on stdout.
    - **Prerequisites for populating databases:**
        - Precomputed asset embeddings and their associated metadata fields for each Milvus collection all stored in multiple JSON files that follow a specific file-hierarchy
        - MongoDB documents of `assetCollections` and `assetsForMetadataExtraction` collections stored in a JSON format

    - **Notice: This script will only work with the newly created Milvus database (without prior data in vector DB) that hasn't been created yet which is acceptable behavior as we don't want to perform this step anytime else but solely at the beginning, as a part of the application setup.**
    - **Notice**: We don't advise using this step unless you're fairly familiar with the project and know the format the data is supposed to have for populating the databases.
    - **Notice**: If you don't see any progress (logs) in the populate-db container, try to list them explicitly in a new terminal while keeping the original terminal with the script running open
    - *This script may take up to 15 minutes.*
1. Execute the following bash script file that deploys all the necessary Docker containers based on the values of the `USE_GPU` and `USE_LLM` ENV variables: `./deploy.sh`


### Stop/Delete the application
If you wish to stop or remove the application, assuming all the previous ENV variables have not been further modified, simply execute the following command: `./deploy.sh --stop` or `./deploy.sh --remove` respectively.

### VM preparations

In order for our application to work properly on a host machine, we need to check whether the following software dependencies are met:
- Docker (test command: `docker ps`)
- CUDA (test command: `nvidia-smi`)
    - CUDA is only necessary for the instances you wish to use a GPU inside the Docker container (`USE_GPU` set to `True`)
- Nvidia toolkit

---

# Repo structure

```
app/                          FastAPI application
├── data/                       Misc data used by FastAPI app
├── models/                     DB entities
├── routers/                    FastAPI endpoints
├── schemas/                    Input/output schemas
├── services/                   Main FastAPI logic
├── config.py                   FastAPI config
└── main.py                     FastAPI entrypoint
experiments/                  Initial experimentation and PoC (not used by FastAPI app)
└── semantic_search/            Semantic search experiments
scripts/                      Helper files and scripts for deployment of FastAPI app
```
