import os
from pathlib import Path
from dotenv import load_dotenv

# Find root directory and optionally load .env file (only exists in local dev, not on HF Spaces)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load .env if it exists (local development), HF Spaces uses Secrets (env vars) directly
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    def __init__(self):
        self.HF_TOKEN: str = os.getenv("HF_TOKEN", "")
        self.GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
        self.IMAGE_MODEL: str = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
        self.GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def is_hf_configured(self) -> bool:
        return bool(self.HF_TOKEN and self.HF_TOKEN != "your_huggingface_token_here")

    def is_groq_configured(self) -> bool:
        return bool(self.GROQ_API_KEY and self.GROQ_API_KEY != "your_groq_api_key_here")

settings = Settings()
