from __future__ import annotations

from functools import lru_cache
from urllib.parse import urljoin

from pydantic import AnyUrl, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

from app_temp.schemas.asset_id import AssetId
from app_temp.schemas.enums import SupportedAssetType


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


class AIoDConfig(BaseModel):
    URL: AnyUrl = Field(...)
    COMMA_SEPARATED_ASSET_TYPES: str = Field(...)
    WINDOW_SIZE: int = Field(1000, le=1000, ge=1)
    WINDOW_OVERLAP: float = Field(0.1, lt=1, ge=0)
    JOB_WAIT_INBETWEEN_REQUESTS_SEC: float = Field(1, ge=0)
    SEARCH_WAIT_INBETWEEN_REQUESTS_SEC: float = Field(0.1, ge=0)
    TESTING: bool = Field(False)
    START_OFFSET: int = Field(0, ge=0)

    @classmethod
    def convert_csv_to_asset_types(cls, value: str) -> list[SupportedAssetType]:
        return [SupportedAssetType(typ) for typ in Validators.validate_csv(value)]

    @field_validator("COMMA_SEPARATED_ASSET_TYPES", mode="before")
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

    def get_assets_url(self, asset_type: SupportedAssetType) -> str:
        return urljoin(str(self.URL), f"{asset_type.value}")

    def get_asset_by_id_url(self, asset_id: AssetId, asset_type: SupportedAssetType) -> str:
        return urljoin(str(self.URL), f"{asset_type.value}/{asset_id}")


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


class Settings(BaseSettings):
    MONGO: MongoConfig = Field(...)
    AIOD: AIoDConfig = Field(...)

    CONNECTION_NUM_RETRIES: int = Field(5, gt=0)
    CONNECTION_SLEEP_TIME: int = Field(30, gt=0)

    class Config:
        env_file = ".env.app"
        env_nested_delimiter = "__"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
