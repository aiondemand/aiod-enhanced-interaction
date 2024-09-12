from functools import lru_cache

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


class MilvusConfig(BaseModel):
    URI: str
    USER: str
    PASS: str
    COLLECTION: str

    TOPK: int = 10
    EMB_DIM: int = 1024
    STORE_CHUNKS: bool = True

    @field_validator("STORE_CHUNKS", mode="before")
    @classmethod
    def str_to_bool(cls, value: str) -> bool:
        if value.lower() == "true":
            return True
        return False


class Settings(BaseSettings):
    MILVUS: MilvusConfig

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = True


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
