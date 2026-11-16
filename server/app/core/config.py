from pathlib import Path

import cloudinary
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class DatabaseSetting(BaseSettings):
    database_url: SecretStr

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        extra="ignore",
        case_sensitive=False,
    )


databasesettings = DatabaseSetting()


class CloudinarySettings(BaseSettings):
    cloudinary_cloud_name: str
    cloudinary_api_key: SecretStr
    cloudinary_api_secret: SecretStr
    cloudinary_secure: bool = True

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("cloudinary_cloud_name", mode="after")
    @classmethod
    def validate_cloud_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Cloudinary cloud name must not be empty or whitespace.")
        return v.strip()

    @field_validator("cloudinary_api_key", mode="after")
    @classmethod
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value() or not v.get_secret_value().strip():
            raise ValueError("Cloudinary API key must not be empty or whitespace.")
        return v

    @field_validator("cloudinary_api_secret", mode="after")
    @classmethod
    def validate_api_secret(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value() or not v.get_secret_value().strip():
            raise ValueError("Cloudinary API secret must not be empty or whitespace.")
        return v


cloudinary_settings = CloudinarySettings()


class OpenAISettings(BaseSettings):
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("openai_model", mode="after")
    @classmethod
    def validate_openai_model(cls, v: str) -> str:
        if not v or not v.strip():
            return "gpt-4o-mini"
        return v.strip()


openai_settings = OpenAISettings()


class LoggingSettings(BaseSettings):
    log_level: str = "INFO"
    nlp_trace_enabled: bool = False
    nlp_trace_content_preview: bool = False
    llm_trace_enabled: bool = False
    llm_trace_full_content: bool = False

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("log_level", mode="after")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        val = v.strip().upper() if v else "INFO"
        return val if val in valid_levels else "INFO"


logging_settings = LoggingSettings()


def configure_cloudinary(settings: CloudinarySettings) -> None:
    """Explicitly configure the Cloudinary Python SDK using the provided settings."""
    cloudinary.config(
        cloud_name=settings.cloudinary_cloud_name,
        api_key=settings.cloudinary_api_key.get_secret_value(),
        api_secret=settings.cloudinary_api_secret.get_secret_value(),
        secure=settings.cloudinary_secure,
    )
