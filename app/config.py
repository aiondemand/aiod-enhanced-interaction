from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin

from pydantic import AnyUrl, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from app.schemas.enums import SupportedAssetType


class Validators:
    @classmethod
    def validate_bool(cls, value: str | bool) -> bool:
        if isinstance(value, bool):
            return value
        elif isinstance(value, str) and value.lower() in ["true", "false"]:
            return value.lower() == "true"
        else:
            raise ValueError("Invalid value for a boolean attribute")


class MilvusConfig(BaseModel):
    URI: AnyUrl = Field(...)
    USER: str = Field(...)
    PASS: str = Field(...)
    COLLECTION_PREFIX: str = Field(..., max_length=100)
    BATCH_SIZE: int = Field(500, gt=0)
    STORE_CHUNKS: bool = Field(True)
    EXTRACT_METADATA: bool = Field(False)
    TIMEOUT: int = Field(60, gt=0)

    @field_validator("COLLECTION_PREFIX", mode="before")
    @classmethod
    def valid_collection_name(cls, value: str):
        if not value[0].isalpha() and value[0] != "_":
            raise ValueError("Collection name must start with a letter or an underscore.")
        if not all(c.isalnum() or c == "_" for c in value):
            raise ValueError("Collection name can only contain letters, numbers, and underscores.")
        return value

    @field_validator("STORE_CHUNKS", "EXTRACT_METADATA", mode="before")
    @classmethod
    def str_to_bool(cls, value: str | bool) -> bool:
        return Validators.validate_bool(value)

    @property
    def MILVUS_TOKEN(self):
        return f"{self.USER}:{self.PASS}"


class OllamaConfig(BaseModel):
    URI: AnyUrl | None = Field(None)
    MODEL_NAME: str = Field("llama3.1:8b", max_length=50)
    NUM_PREDICT: int = Field(1_024, gt=0)
    NUM_CTX: int = Field(4_096, gt=0)
    TIMEOUT: int = Field(120, gt=0)


class AIoDConfig(BaseModel):
    URL: AnyUrl = Field(...)
    COMMA_SEPARATED_ASSET_TYPES: str = Field(...)
    COMMA_SEPARATED_ASSET_TYPES_FOR_METADATA_EXTRACTION: str = Field(...)
    WINDOW_SIZE: int = Field(1000, le=1000, gt=1)
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
        types = value.lower().split(",")
        return [SupportedAssetType(typ.strip()) for typ in types if len(typ.strip()) > 0]

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
                "AIoD assets for metadata extraction is not a subset of all AIoD assets we support"
            )
        return types

    def get_assets_url(self, asset_type: SupportedAssetType) -> str:
        return urljoin(str(self.URL), f"{asset_type.value}/v1")

    def get_asset_by_id_url(self, asset_id: int, asset_type: SupportedAssetType) -> str:
        return urljoin(str(self.URL), f"{asset_type.value}/v1/{asset_id}")


class MongoConfig(BaseModel):
    URI: AnyUrl = Field(...)
    DBNAME: str = Field("aiod")


class Settings(BaseSettings):
    MILVUS: MilvusConfig = Field(...)
    MONGODB: MongoConfig = Field(...)
    AIOD: AIoDConfig = Field(...)
    OLLAMA: OllamaConfig = Field(...)

    USE_GPU: bool = Field(False)
    MODEL_LOADPATH: str = Field(...)
    MODEL_BATCH_SIZE: int = Field(..., gt=0)
    CONNECTION_NUM_RETRIES: int = Field(5, gt=0)
    CONNECTION_SLEEP_TIME: int = Field(30, gt=0)
    QUERY_EXPIRATION_TIME_IN_MINUTES: int = Field(10, gt=0)

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

    @property
    def PERFORM_LLM_QUERY_PARSING(self) -> bool:
        return (
            self.MILVUS.EXTRACT_METADATA
            and self.OLLAMA.URI is not None
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
