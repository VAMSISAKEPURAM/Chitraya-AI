import os
from pathlib import Path
from dotenv import load_dotenv

# Find root directory and optionally load .env file (for local dev)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    @property
    def HF_TOKEN(self) -> str:
        return (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
            or os.getenv("HF_API_TOKEN")
            or ""
        ).strip()

    @property
    def GROQ_API_KEY(self) -> str:
        return (
            os.getenv("GROQ_API_KEY")
            or os.getenv("GROQ_TOKEN")
            or ""
        ).strip()

    @property
    def IMAGE_MODEL(self) -> str:
        return os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell").strip()

    @property
    def GROQ_MODEL(self) -> str:
        return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

    def is_hf_configured(self) -> bool:
        token = self.HF_TOKEN
        return bool(token and token != "your_huggingface_token_here")

    def is_groq_configured(self) -> bool:
        key = self.GROQ_API_KEY
        return bool(key and key != "your_groq_api_key_here")

settings = Settings()
