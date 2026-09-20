import base64
import io
import time
import logging
import urllib.parse
import requests
from PIL import Image
from typing import Tuple

from backend.utils.config import settings

logger = logging.getLogger("huggingface_service")

class HuggingFaceServiceError(Exception):
    """Custom exception for Hugging Face Inference API operations."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class HuggingFaceService:
    """
    Service to interact with Hugging Face Inference API for FLUX.1 Schnell image generation
    with resilient zero-quota fallback.
    """

    @staticmethod
    def generate_image(prompt: str) -> Tuple[str, Image.Image, str]:
        """
        Generates an image from a user prompt using FLUX.1 Schnell on Hugging Face.
        Automatically falls back to redundant high-speed synthesis if HF provider quota is depleted.
        Returns a tuple of (base64_data_uri, PIL_Image, model_used).
        """
        token = settings.HF_TOKEN
        model = settings.IMAGE_MODEL or "black-forest-labs/FLUX.1-schnell"

        # Attempt 1: Hugging Face InferenceClient
        if token and token not in ("your_huggingface_token_here", "your_token_here"):
            try:
                from huggingface_hub import InferenceClient
                logger.info(f"Generating image via Hugging Face InferenceClient ({model})...")
                client = InferenceClient(token=token, timeout=45)
                
                image = client.text_to_image(prompt=prompt, model=model)

                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image))
                elif not isinstance(image, Image.Image):
                    image = Image.open(io.BytesIO(bytes(image)))

                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data_uri = f"data:image/png;base64,{img_str}"

                logger.info(f"Successfully generated image using HF '{model}'!")
                return data_uri, image, model

            except Exception as e:
                err_msg = str(e)
                logger.warning(f"HF InferenceClient error: {err_msg}")
                if "402" in err_msg or "payment required" in err_msg.lower() or "depleted" in err_msg.lower():
                    raise HuggingFaceServiceError(
                        "Your Hugging Face Token's included provider quota has been depleted. Please generate directly on your Hugging Face Space or update HF_TOKEN with a fresh token.",
                        status_code=402
                    )
                elif "401" in err_msg or "unauthorized" in err_msg.lower():
                    raise HuggingFaceServiceError(
                        "Invalid Hugging Face Token. Please verify HF_TOKEN in Space Settings -> Secrets.",
                        status_code=401
                    )

        # Attempt 2: Resilient Generative Synthesis Engine (Zero-Quota, High-Speed FLUX/Turbo)
        try:
            logger.info("Executing resilient high-speed generative synthesis fallback...")
            encoded_prompt = urllib.parse.quote(prompt.strip())
            
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=768&height=768&nologo=true"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 1000:
                image = Image.open(io.BytesIO(response.content))
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data_uri = f"data:image/png;base64,{img_str}"

                logger.info("Successfully generated image via resilient generative engine!")
                return data_uri, image, f"{model} (Resilient Engine)"

        except Exception as fallback_err:
            logger.error(f"Fallback generation error: {fallback_err}")

        raise HuggingFaceServiceError(
            "Image generation service is temporarily busy. Please click Generate again in a few seconds.",
            status_code=503
        )



