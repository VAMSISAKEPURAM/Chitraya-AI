import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    """Dynamic configuration loader with multi-alias support for environment secrets."""

    def reload_env(self) -> None:
        """Reload .env file to pick up any changes without restarting the process."""
        if ENV_PATH.exists():
            load_dotenv(dotenv_path=ENV_PATH, override=True)

    @property
    def HF_TOKEN(self) -> str:
        token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or os.getenv("HF_API_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or ""
        ).strip().strip("\"'")
        return token

    @property
    def GROQ_API_KEY(self) -> str:
        key = (
            os.getenv("GROQ_API_KEY")
            or os.getenv("GROQ_TOKEN")
            or ""
        ).strip().strip("\"'")
        return key

    @property
    def IMAGE_MODEL(self) -> str:
        return os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell").strip().strip("\"'")

    @property
    def GROQ_MODEL(self) -> str:
        return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip().strip("\"'")

    def is_hf_configured(self) -> bool:
        token = self.HF_TOKEN
        # If not configured in memory, attempt a reload from .env in case user updated it
        if not token or token in ("your_huggingface_token_here", "your_token_here"):
            self.reload_env()
            token = self.HF_TOKEN
        return bool(
            token
            and token not in ("your_huggingface_token_here", "your_token_here")
            and not token.startswith("<")
        )

    def is_groq_configured(self) -> bool:
        key = self.GROQ_API_KEY
        if not key or key in ("your_groq_api_key_here", "your_key_here"):
            self.reload_env()
            key = self.GROQ_API_KEY
        return bool(
            key
            and key not in ("your_groq_api_key_here", "your_key_here")
            and not key.startswith("<")
        )

settings = Settings()

