from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import cast
from urllib.parse import urljoin

from pydantic import AnyUrl, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from app.schemas.asset_id import AssetId
from app.schemas.enums import SupportedAssetType


SUPPORTED_ASSET_TYPES_FOR_METADATA_FILTERING = [
    SupportedAssetType.DATASETS,
    SupportedAssetType.ML_MODELS,
    SupportedAssetType.PUBLICATIONS,
    SupportedAssetType.EDUCATIONAL_RESOURCES,
]


class Validators:
    @classmethod
    def validate_bool(cls, value: str | bool) -> bool:
        if isinstance(value, bool):
            return value
        elif isinstance(value, str) and value.lower() in ["true", "false"]:
            return value.lower() == "true"
        else:
            raise ValueError("Invalid value for a boolean attribute")

    @classmethod
    def validate_csv(cls, value: str) -> list[str]:
        values = value.lower().split(",")
        return [val.strip() for val in values if len(val.strip()) > 0]


class MilvusConfig(BaseModel):
    USE_LITE: bool = Field(default=False)
    FILEPATH: Path | None = Field(default=None)  # specify path to the file-based Milvus Lite DB
    URI: AnyUrl = Field(...)
    USER: str = Field(...)
    PASS: str = Field(...)
    COLLECTION_PREFIX: str = Field(..., max_length=100)
    BATCH_SIZE: int = Field(500, gt=0)
    STORE_CHUNKS: bool = Field(True)
    TIMEOUT: int = Field(60, gt=0)

    @field_validator("COLLECTION_PREFIX", mode="before")
    @classmethod
    def valid_collection_name(cls, value: str):
        if not value[0].isalpha() and value[0] != "_":
            raise ValueError("Collection name must start with a letter or an underscore.")
        if not all(c.isalnum() or c == "_" for c in value):
            raise ValueError("Collection name can only contain letters, numbers, and underscores.")
        return value

    @field_validator("STORE_CHUNKS", "USE_LITE", mode="before")
    @classmethod
    def str_to_bool(cls, value: str | bool) -> bool:
        return Validators.validate_bool(value)

    @model_validator(mode="after")
    def check_lite(self) -> MilvusConfig:
        if self.USE_LITE and self.FILEPATH is None:
            raise ValueError("You need to specify the path to Milvus Lite DB file.")

        return self

    @property
    def HOST(self) -> str:
        return str(self.FILEPATH) if self.USE_LITE else str(self.URI)

    @property
    def MILVUS_TOKEN(self) -> str:
        return "" if self.USE_LITE else f"{self.USER}:{self.PASS}"


class MetadataFilteringConfig(BaseModel):
    ENABLED: bool = Field(True)
    # Whether we wish to run a subagent checking values adhere to fields' associated enums
    # For both the metadata extraction & user query parsing
    ENFORCE_ENUMS: bool = Field(True)

    @field_validator("ENABLED", "ENFORCE_ENUMS", mode="before")
    @classmethod
    def str_to_bool(cls, value: str | bool) -> bool:
        return Validators.validate_bool(value)


class CrawlerConfig(BaseModel):
    COMMA_SEPARATED_WEBSITES: str = Field(...)
    COMMA_SEPARATED_API_WEBSITES: str = Field(...)
    COMMA_SEPARATED_BLOCKED_WEBSITES: str = Field(...)

    @classmethod
    def convert_csv_to_urls(cls, value: str, to_string: bool) -> list[AnyUrl] | list[str]:
        websites = Validators.validate_csv(value)
        if to_string:
            return websites
        else:
            return [AnyUrl(url) for url in websites]

    @field_validator(
        "COMMA_SEPARATED_WEBSITES",
        "COMMA_SEPARATED_API_WEBSITES",
        "COMMA_SEPARATED_BLOCKED_WEBSITES",
        mode="before",
    )
    @classmethod
    def validate_csv_urls(cls, value: str) -> str:
        try:
            cls.convert_csv_to_urls(value, to_string=False)
        except ValueError:
            ValueError("Invalid asset types defined")
        return value

    @property
    def WEBSITES(self) -> list[str]:
        return cast(
            list[str], self.convert_csv_to_urls(self.COMMA_SEPARATED_WEBSITES, to_string=True)
        )

    @property
    def API_WEBSITES(self) -> list[str]:
        return cast(
            list[str], self.convert_csv_to_urls(self.COMMA_SEPARATED_API_WEBSITES, to_string=True)
        )

    @property
    def BLOCKED_WEBSITES(self) -> list[str]:
        return cast(
            list[str],
            self.convert_csv_to_urls(self.COMMA_SEPARATED_BLOCKED_WEBSITES, to_string=True),
        )


class OllamaConfig(BaseModel):
    URI: AnyUrl | None = Field(None)
    MODEL_NAME: str = Field("qwen3:4b-instruct")
    MAX_TOKENS: int = Field(1_024, gt=0)


class AIoDConfig(BaseModel):
    URL: AnyUrl = Field(...)
    COMMA_SEPARATED_ASSET_TYPES: str = Field(...)
    COMMA_SEPARATED_ASSET_TYPES_FOR_METADATA_EXTRACTION: str = Field(...)
    WINDOW_SIZE: int = Field(1000, le=1000, ge=1)
    WINDOW_OVERLAP: float = Field(0.1, lt=1, ge=0)
    JOB_WAIT_INBETWEEN_REQUESTS_SEC: float = Field(1, ge=0)
    SEARCH_WAIT_INBETWEEN_REQUESTS_SEC: float = Field(0.1, ge=0)

    DAY_IN_MONTH_FOR_EMB_CLEANING: int = Field(1, ge=1, le=31)
    DAY_IN_MONTH_FOR_TRAVERSING_ALL_AIOD_ASSETS: int = Field(5, ge=1, le=31)
    TESTING: bool = Field(False)
    STORE_DATA_IN_JSON: bool = Field(False)
    JSON_SAVEPATH: Path | None = Field(None)

    @classmethod
    def convert_csv_to_asset_types(cls, value: str) -> list[SupportedAssetType]:
        return [SupportedAssetType(typ) for typ in Validators.validate_csv(value)]

    @field_validator(
        "COMMA_SEPARATED_ASSET_TYPES",
        "COMMA_SEPARATED_ASSET_TYPES_FOR_METADATA_EXTRACTION",
        mode="before",
    )
    @classmethod
    def validate_asset_types(cls, value: str) -> str:
        try:
            cls.convert_csv_to_asset_types(value)
        except ValueError:
            ValueError("Invalid asset types defined")
        return value

    @field_validator("TESTING", mode="before")
    @classmethod
    def validate_bool(cls, value: str | bool) -> bool:
        return Validators.validate_bool(value)

    @model_validator(mode="after")
    def check_json_savepath(self) -> AIoDConfig:
        if self.STORE_DATA_IN_JSON and self.JSON_SAVEPATH is None:
            raise ValueError(
                "You need to specify 'JSON_SAVEPATH' env var if 'STORE_DATA_IN_JSON' env var is set to True."
            )
        return self

    @model_validator(mode="after")
    def change_window_size(self) -> AIoDConfig:
        if self.TESTING:
            self.WINDOW_SIZE = 10
        return self

    @property
    def OFFSET_INCREMENT(self) -> int:
        return int(settings.AIOD.WINDOW_SIZE * (1 - settings.AIOD.WINDOW_OVERLAP))

    @property
    def ASSET_TYPES(self) -> list[SupportedAssetType]:
        return self.convert_csv_to_asset_types(self.COMMA_SEPARATED_ASSET_TYPES)

    @property
    def ASSET_TYPES_FOR_METADATA_EXTRACTION(self) -> list[SupportedAssetType]:
        types = self.convert_csv_to_asset_types(
            self.COMMA_SEPARATED_ASSET_TYPES_FOR_METADATA_EXTRACTION
        )

        if not set(types).issubset(set(self.ASSET_TYPES)):
            raise ValueError(
                "AIoD assets for metadata extraction (env var: 'AIOD__COMMA_SEPARATED_ASSET_TYPES_FOR_METADATA_EXTRACTION') "
                + "is not a subset of all AIoD assets we support (env var: 'AIOD__COMMA_SEPARATED_ASSET_TYPES')"
            )
        if not set(types).issubset(set(SUPPORTED_ASSET_TYPES_FOR_METADATA_FILTERING)):
            diff = set(types) - set(SUPPORTED_ASSET_TYPES_FOR_METADATA_FILTERING)

            raise ValueError(
                f"We DO NOT support the following asset types for metadata extraction: {[asset.value for asset in diff]}"
            )

        return types

    def get_assets_url(self, asset_type: SupportedAssetType) -> str:
        return urljoin(str(self.URL), f"{asset_type.value}")

    def get_asset_by_id_url(self, asset_id: AssetId, asset_type: SupportedAssetType) -> str:
        return urljoin(str(self.URL), f"{asset_type.value}/{asset_id}")

    def get_taxonomy_url(self, taxonomy: str) -> str:
        return urljoin(str(self.URL), f"{taxonomy}")


class MongoConfig(BaseModel):
    HOST: str = Field(...)
    PORT: int = Field(...)
    DBNAME: str = Field("aiod")
    AUTH_DBNAME: str = Field("admin")
    USER: str = Field(...)
    PASSWORD: str = Field(...)

    @property
    def CONNECTION_STRING(self) -> str:
        return f"mongodb://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DBNAME}?authSource={self.AUTH_DBNAME}"


class ChatbotConfig(BaseModel):
    USE_CHATBOT: bool = Field(...)
    MISTRAL_KEY: str = Field(...)
    MISTRAL_MODEL: str = Field("mistral-medium-latest")
    TOP_K_ASSETS_TO_SEARCH: int = Field(10, gt=0)
    MYLIBRARY_URL: AnyUrl = Field(...)

    @property
    def WEBSITE_COLLECTION_NAME(self) -> str:
        return "website_collection"

    @property
    def API_COLLECTION_NAME(self) -> str:
        return "api_collection"

    @field_validator("USE_CHATBOT", mode="before")
    @classmethod
    def str_to_bool(cls, value: str | bool) -> bool:
        return Validators.validate_bool(value)

    def generate_mylibrary_asset_url(self, asset_id: str, asset_type: SupportedAssetType) -> str:
        _mapping = {
            SupportedAssetType.DATASETS: "Dataset",
            SupportedAssetType.ML_MODELS: "AIModel",
            SupportedAssetType.PUBLICATIONS: "Publication",
            SupportedAssetType.CASE_STUDIES: r"Success%stories",
            SupportedAssetType.EDUCATIONAL_RESOURCES: r"Educational%20resource",
            SupportedAssetType.EXPERIMENTS: "Experiment",
            SupportedAssetType.SERVICES: "Service",
            SupportedAssetType.COMPUTATIONAL_ASSETS: r"Computational%20asset",
            SupportedAssetType.RESOURCE_BUNDLES: r"Resource%20Bundle",
        }

        return urljoin(
            str(settings.CHATBOT.MYLIBRARY_URL),
            f"resources/{asset_id}?category={_mapping[asset_type]}",
        )


class Settings(BaseSettings):
    MILVUS: MilvusConfig = Field(...)
    MONGO: MongoConfig = Field(...)
    AIOD: AIoDConfig = Field(...)
    METADATA_FILTERING: MetadataFilteringConfig = Field(default=MetadataFilteringConfig())
    OLLAMA: OllamaConfig = Field(default=OllamaConfig())
    CHATBOT: ChatbotConfig = Field(...)
    CRAWLER: CrawlerConfig = Field(...)

    API_VERSION: str = Field(...)
    USE_GPU: bool = Field(False)
    MODEL_LOADPATH: str = Field(...)
    MODEL_BATCH_SIZE: int = Field(..., gt=0)
    CONNECTION_NUM_RETRIES: int = Field(5, gt=0)
    CONNECTION_SLEEP_TIME: int = Field(30, gt=0)
    QUERY_EXPIRATION_TIME_IN_MINUTES: int = Field(10, gt=0)

    LOGFIRE_TOKEN: str | None = Field(None)

    @field_validator("USE_GPU", mode="before")
    @classmethod
    def validate_bool(cls, value: str | bool) -> bool:
        return Validators.validate_bool(value)

    @field_validator("MODEL_LOADPATH", mode="before")
    @classmethod
    def validate_model_loadpath(cls, value: str) -> str:
        path = Path(value)

        if value == "Alibaba-NLP/gte-large-en-v1.5" or path.exists():
            return value
        raise ValueError("Invalid loadpath for the model.")

    @model_validator(mode="after")
    def check_milvus_lite(self) -> Settings:
        if self.MILVUS.USE_LITE and self.METADATA_FILTERING.ENABLED:
            raise ValueError("We don't support Metadata Filtering when using Milvus Lite")
        else:
            return self

    @property
    def USING_OLLAMA(self) -> bool:
        return self.OLLAMA.URI is not None

    def extracts_metadata_from_asset(self, asset_type: SupportedAssetType) -> bool:
        return (
            self.MILVUS.USE_LITE is False
            and asset_type in self.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION
        )

    @property
    def PERFORM_METADATA_EXTRACTION(self) -> bool:
        return self.USING_OLLAMA and self.METADATA_FILTERING.ENABLED

    @property
    def PERFORM_LLM_QUERY_PARSING(self) -> bool:
        return (
            self.PERFORM_METADATA_EXTRACTION
            and len(self.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION) > 0
        )

    class Config:
        env_file = ".env.app"
        env_nested_delimiter = "__"
        case_sensitive = True


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
