from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin

from app.schemas.enums import AssetType
from pydantic import AnyUrl, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class Validators:
    @classmethod
    def str_to_bool(cls, value: str) -> bool:
        if value.lower() not in ["true", "false"]:
            raise ValueError("Invalid value for boolean attribute")
        return value.lower() == "true"


class MilvusConfig(BaseModel):
    URI: AnyUrl = Field(...)
    USER: str = Field(...)
    PASS: str = Field(...)
    COLLECTION_PREFIX: str = Field(..., max_length=100)
    BATCH_SIZE: int = Field(500, gt=0)
    STORE_CHUNKS: bool = Field(True)
    EXTRACT_METADATA: bool = Field(False)

    @field_validator("COLLECTION_PREFIX", mode="before")
    @classmethod
    def valid_collection_name(cls, value: str):
        if not value[0].isalpha() and value[0] != "_":
            raise ValueError(
                "Collection name must start with a letter or an underscore."
            )
        if not all(c.isalnum() or c == "_" for c in value):
            raise ValueError(
                "Collection name can only contain letters, numbers, and underscores."
            )
        return value

    @field_validator("STORE_CHUNKS", "EXTRACT_METADATA", mode="before")
    @classmethod
    def str_to_bool(cls, value: str) -> bool:
        return Validators.str_to_bool(value)

    @property
    def MILVUS_TOKEN(self):
        return f"{self.USER}:{self.PASS}"


class OllamaConfig(BaseModel):
    URI: AnyUrl | None = Field(None)
    MODEL_NAME: str = Field("llama3.1:8b", max_length=50)
    NUM_PREDICT: int = Field(1_024, gt=0)
    NUM_CTX: int = Field(4_096, gt=0)


class AIoDConfig(BaseModel):
    URL: AnyUrl = Field(...)
    COMMA_SEPARETED_ASSET_TYPES: str = Field(...)
    COMMA_SEPARATED_ASSET_TYPES_FOR_METADATA_EXTRACTON: str = Field(...)
    WINDOW_SIZE: int = Field(1000, le=1000, gt=1)
    WINDOW_OVERLAP: float = Field(0.1, lt=1, ge=0)
    JOB_WAIT_INBETWEEN_REQUESTS_SEC: float = Field(1, ge=0)
    SEARCH_WAIT_INBETWEEN_REQUESTS_SEC: float = Field(0.1, ge=0)

    DAY_IN_MONTH_FOR_EMB_CLEANING: int = Field(1, ge=1, le=31)
    DAY_IN_MONTH_FOR_TRAVERSING_ALL_AIOD_ASSETS: int = Field(5, ge=1, le=31)
    TESTING: bool = Field(False)

    @field_validator(
        "COMMA_SEPARETED_ASSET_TYPES",
        "COMMA_SEPARATED_ASSET_TYPES_FOR_METADATA_EXTRACTON",
        mode="before",
    )
    @classmethod
    def validate_asset_types(cls, value: str) -> str:
        try:
            types = value.lower().split("")
            types = [AssetType(typ) for typ in types]
        except ValueError:
            ValueError("Invalid asset types defined")
        return value

    @field_validator("TESTING", mode="before")
    @classmethod
    def str_to_bool(cls, value: str) -> bool:
        return Validators.str_to_bool(value)

    @property
    def OFFSET_INCREMENT(self) -> int:
        return int(settings.AIOD.WINDOW_SIZE * (1 - settings.AIOD.WINDOW_OVERLAP))

    @property
    def ASSET_TYPES(self) -> list[str]:
        types = self.COMMA_SEPARETED_ASSET_TYPES.lower().split(",")
        return [AssetType(typ) for typ in types]

    @property
    def ASSET_TYPES_FOR_METADATA_EXTRACTION(self) -> list[str]:
        types = self.COMMA_SEPARATED_ASSET_TYPES_FOR_METADATA_EXTRACTON.lower().split(
            ","
        )
        types = [AssetType(typ) for typ in types]

        assert set(types).issubset(
            set(self.ASSET_TYPES)
        ), "AIoD assets for metadata extraction is not a subset of all AIoD assets we support"
        return types

    def get_assets_url(self, asset_type: AssetType) -> str:
        return urljoin(str(self.URL), f"{asset_type.value}/v1")

    def get_asset_by_id_url(self, doc_id: str, asset_type: AssetType) -> str:
        return urljoin(str(self.URL), f"{asset_type.value}/v1/{doc_id}")


class Settings(BaseSettings):
    MILVUS: MilvusConfig = Field(...)
    AIOD: AIoDConfig = Field(...)
    OLLAMA: OllamaConfig = Field(...)

    USE_GPU: bool = Field(False)
    TINYDB_FILEPATH: Path = Field(...)
    MODEL_LOADPATH: str = Field(...)
    MODEL_BATCH_SIZE: int = Field(..., gt=0)

    @field_validator("USE_GPU", mode="before")
    @classmethod
    def str_to_bool(cls, value: str) -> bool:
        return Validators.str_to_bool(value)

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
