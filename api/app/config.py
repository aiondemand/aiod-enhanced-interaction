from functools import lru_cache

from app.schemas.enums import AssetType
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


class MilvusConfig(BaseModel):
    URI: str
    USER: str
    PASS: str
    COLLECTION_PREFIX: str

    TOPK: int = 10
    EMB_DIM: int = 1024
    BATCH_SIZE: int = 500
    STORE_CHUNKS: bool = True

    @field_validator("STORE_CHUNKS", mode="before")
    @classmethod
    def str_to_bool(cls, value: str) -> bool:
        if value.lower() == "true":
            return True
        return False

    @property
    def MILVUS_TOKEN(self):
        return f"{self.USER}:{self.PASS}"

    def get_collection_name(self, asset_type: AssetType):
        return f"{self.COLLECTION_PREFIX}_{asset_type.value}"


class AIoDConfig(BaseModel):
    URL: str
    COMMA_SEPARETED_ASSET_TYPES: str  # TODO validator needed
    WINDOW_SIZE: int = 1000
    TIMEOUT_REQUEST_INTERVAL_SEC: int = 3

    @field_validator("URL", mode="before")
    @classmethod
    def remove_trailing_slash(cls, value: str) -> str:
        return value.strip("/")

    @property
    def ASSET_TYPES(self) -> list[str]:
        types = self.COMMA_SEPARETED_ASSET_TYPES.lower().split(",")
        return [AssetType(typ) for typ in types]

    def get_asset_url(self, asset_type: AssetType) -> str:
        return f"{self.URL}/{asset_type.value}/v1"

    def get_asset_count_url(self, asset_type: AssetType) -> str:
        return f"{self.URL}/counts/{asset_type.value}/v1"


class Settings(BaseSettings):
    MILVUS: MilvusConfig
    AIOD: AIoDConfig
    USE_GPU: bool = False
    TINYDB_FILEPATH: str
    MODEL_BATCH_SIZE: int

    @field_validator("USE_GPU", mode="before")
    @classmethod
    def str_to_bool(cls, value: str) -> bool:
        if value.lower() == "true":
            return True
        return False

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = True


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
