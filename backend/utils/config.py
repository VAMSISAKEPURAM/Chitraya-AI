import os
from pathlib import Path
from dotenv import load_dotenv

# Find root directory and load .env file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    IMAGE_MODEL: str = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    @classmethod
    def is_hf_configured(cls) -> bool:
        return bool(cls.HF_TOKEN and cls.HF_TOKEN != "your_huggingface_token_here")

    @classmethod
    def is_groq_configured(cls) -> bool:
        return bool(cls.GROQ_API_KEY and cls.GROQ_API_KEY != "your_groq_api_key_here")

settings = Settings()
