"""Tests for Cloudinary configuration and credential validation.

Verifies:
1. Environment loading from .env via Pydantic Settings.
2. Credential validation against empty and whitespace-only strings.
3. Secret masking in representations (no secret credential exposure).
4. Explicit configure_cloudinary helper invoking cloudinary.config with secure=True.
5. Startup invocation within FastAPI lifespan.
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import SecretStr, ValidationError
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import app.core.config as config_module
from app.core.config import (
    CloudinarySettings,
    cloudinary_settings,
    configure_cloudinary,
)
from app.main import app, lifespan


class TestCloudinarySettings:
    """Unit tests for CloudinarySettings model and validation."""

    def test_settings_loaded_from_environment(self) -> None:
        """CloudinarySettings successfully loads credentials from .env."""
        assert isinstance(cloudinary_settings.cloudinary_cloud_name, str)
        assert len(cloudinary_settings.cloudinary_cloud_name) > 0

        assert isinstance(cloudinary_settings.cloudinary_api_key, SecretStr)
        assert len(cloudinary_settings.cloudinary_api_key.get_secret_value()) > 0

        assert isinstance(cloudinary_settings.cloudinary_api_secret, SecretStr)
        assert len(cloudinary_settings.cloudinary_api_secret.get_secret_value()) > 0

        assert cloudinary_settings.cloudinary_secure is True

    @pytest.mark.parametrize(
        "invalid_cloud_name",
        ["", "   ", "\t", "\n"],
    )
    def test_cloud_name_empty_or_whitespace_rejected(self, invalid_cloud_name: str) -> None:
        """Empty or whitespace-only cloud names must be rejected."""
        with pytest.raises(ValidationError, match="Cloudinary cloud name must not be empty or whitespace"):
            CloudinarySettings(
                cloudinary_cloud_name=invalid_cloud_name,
                cloudinary_api_key=SecretStr("valid_key"),
                cloudinary_api_secret=SecretStr("valid_secret"),
                cloudinary_secure=True,
            )

    @pytest.mark.parametrize(
        "invalid_api_key",
        ["", "   ", "\t", "\n"],
    )
    def test_api_key_empty_or_whitespace_rejected(self, invalid_api_key: str) -> None:
        """Empty or whitespace-only API keys must be rejected."""
        with pytest.raises(ValidationError, match="Cloudinary API key must not be empty or whitespace"):
            CloudinarySettings(
                cloudinary_cloud_name="valid_cloud",
                cloudinary_api_key=SecretStr(invalid_api_key),
                cloudinary_api_secret=SecretStr("valid_secret"),
                cloudinary_secure=True,
            )

    @pytest.mark.parametrize(
        "invalid_api_secret",
        ["", "   ", "\t", "\n"],
    )
    def test_api_secret_empty_or_whitespace_rejected(self, invalid_api_secret: str) -> None:
        """Empty or whitespace-only API secrets must be rejected."""
        with pytest.raises(ValidationError, match="Cloudinary API secret must not be empty or whitespace"):
            CloudinarySettings(
                cloudinary_cloud_name="valid_cloud",
                cloudinary_api_key=SecretStr("valid_key"),
                cloudinary_api_secret=SecretStr(invalid_api_secret),
                cloudinary_secure=True,
            )

    def test_secret_masking_in_repr_and_str(self) -> None:
        """Sensitive credentials must never be exposed in repr or str output."""
        custom_settings = CloudinarySettings(
            cloudinary_cloud_name="masked_cloud",
            cloudinary_api_key=SecretStr("super_secret_api_key_123"),
            cloudinary_api_secret=SecretStr("super_secret_api_secret_456"),
            cloudinary_secure=True,
        )

        settings_repr = repr(custom_settings)
        settings_str = str(custom_settings)

        assert "super_secret_api_key_123" not in settings_repr
        assert "super_secret_api_secret_456" not in settings_repr
        assert "super_secret_api_key_123" not in settings_str
        assert "super_secret_api_secret_456" not in settings_str
        assert "**********" in settings_repr

    def test_secure_defaults_to_true(self) -> None:
        """Secure setting defaults to True if omitted."""
        custom_settings = CloudinarySettings(
            cloudinary_cloud_name="custom_cloud",
            cloudinary_api_key=SecretStr("custom_key"),
            cloudinary_api_secret=SecretStr("custom_secret"),
        )
        assert custom_settings.cloudinary_secure is True


class TestCloudinarySDKConfiguration:
    """Tests for Cloudinary SDK configuration helper and lifecycle invocation."""

    def test_configure_cloudinary_invokes_sdk_with_unpacked_secrets(self) -> None:
        """configure_cloudinary correctly passes unpacked secrets and secure flag to cloudinary.config."""
        custom_settings = CloudinarySettings(
            cloudinary_cloud_name="my_test_cloud",
            cloudinary_api_key=SecretStr("mock_api_key"),
            cloudinary_api_secret=SecretStr("mock_api_secret"),
            cloudinary_secure=True,
        )

        with patch("cloudinary.config") as mock_config:
            configure_cloudinary(custom_settings)
            mock_config.assert_called_once_with(
                cloud_name="my_test_cloud",
                api_key="mock_api_key",
                api_secret="mock_api_secret",
                secure=True,
            )

    def test_importing_config_does_not_invoke_cloudinary_config(self) -> None:
        """Importing app.core.config must NOT configure the Cloudinary SDK automatically."""
        with patch("cloudinary.config") as mock_config:
            importlib.reload(config_module)
            mock_config.assert_not_called()

    @pytest.mark.asyncio
    async def test_lifespan_startup_invokes_configure_cloudinary(self) -> None:
        """FastAPI lifespan startup explicitly invokes configure_cloudinary with cloudinary_settings."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__.return_value = mock_session
        mock_session_ctx.__aexit__.return_value = None

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch("app.main.AsyncSessionLocal", return_value=mock_session_ctx),
            patch("app.main.engine", mock_engine),
            patch("app.main.configure_cloudinary") as mock_configure,
        ):
            async with lifespan(app):
                mock_configure.assert_called_once_with(cloudinary_settings)
