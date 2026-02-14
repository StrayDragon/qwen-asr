"""
Configuration management for OpenAI-compatible server.
"""

import os
from pathlib import Path
from typing import Dict


class Settings:
    """Global settings and configuration."""

    # Model name -> directory mapping
    MODELS: Dict[str, str] = {
        "qwen-asr-0.6b": "qwen3-asr-0.6b",
        "qwen-asr-1.7b": "qwen3-asr-1.7b",
    }

    # Default model (0.6b is typically more available)
    DEFAULT_MODEL: str = "qwen-asr-0.6b"

    # Model pool configuration
    MODEL_POOL_SIZE: int = int(os.getenv("QWEN_MODEL_POOL_SIZE", "2"))

    # Model auto-unload after idle timeout (seconds)
    # Set to 0 to disable auto-unload
    MODEL_IDLE_TIMEOUT: int = int(
        os.getenv("QWEN_MODEL_IDLE_TIMEOUT", "600")
    )  # 10 minutes default

    # Authentication
    API_TOKEN: str = os.getenv("QWEN_API_TOKEN", "sk-test-key")

    # Server configuration
    HOST: str = os.getenv("QWEN_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("QWEN_PORT", "8011"))

    @staticmethod
    def get_available_models() -> list[str]:
        """Get list of available model names.

        Checks if model directory contains model.safetensors manifest.
        """
        base_dir = Path(__file__).parent.parent
        available = []

        for model_name, model_dir in Settings.MODELS.items():
            model_path = base_dir / model_dir
            manifest_path = model_path / "model.safetensors"

            if manifest_path.exists():
                available.append(model_name)

        return available

    @staticmethod
    def get_model_dir(model_name: str) -> str:
        """Resolve model name to absolute directory path.

        Hardcoded to qwen-asr-0.6b for Docker deployment.
        """
        base_dir = Path(__file__).parent.parent
        return str(base_dir / "qwen3-asr-0.6b")


# Global settings instance
settings = Settings()
