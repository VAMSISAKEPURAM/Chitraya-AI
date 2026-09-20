import base64
import io
import time
import logging
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
    Service to interact with Hugging Face Inference API for FLUX.1 Schnell image generation.
    Uses official InferenceClient with automatic exponential retry.
    """

    @staticmethod
    def generate_image(prompt: str) -> Tuple[str, Image.Image, str]:
        """
        Generates an image from a user prompt using FLUX.1 Schnell on Hugging Face.
        Automatically retries on temporary serverless congestion.
        Returns a tuple of (base64_data_uri, PIL_Image, model_used).
        """
        if not settings.is_hf_configured():
            raise HuggingFaceServiceError(
                "Hugging Face API Token (HF_TOKEN) is not configured. Please add your token in Space Settings -> Secrets.",
                status_code=401
            )

        token = settings.HF_TOKEN
        model = settings.IMAGE_MODEL or "black-forest-labs/FLUX.1-schnell"

        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token, timeout=60)

        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Generating image with '{model}' (Attempt {attempt}/{max_retries})...")
                
                # Invoke FLUX.1 Schnell text-to-image via official InferenceClient
                image = client.text_to_image(prompt=prompt, model=model)

                # Ensure image is a valid PIL Image
                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image))
                elif not isinstance(image, Image.Image):
                    image = Image.open(io.BytesIO(bytes(image)))

                # Convert to base64 data URI
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data_uri = f"data:image/png;base64,{img_str}"

                logger.info(f"Successfully generated image using '{model}' on attempt {attempt}!")
                return data_uri, image, model

            except Exception as e:
                last_error = e
                err_msg = str(e)
                logger.warning(f"InferenceClient attempt {attempt} failed: {err_msg}")

                if "401" in err_msg or "unauthorized" in err_msg.lower():
                    raise HuggingFaceServiceError(
                        "Invalid or unauthorized Hugging Face token. Please check HF_TOKEN in Space Settings -> Secrets.",
                        status_code=401
                    )

                # If serverless model is loading or congested, wait and retry
                if attempt < max_retries:
                    wait_time = attempt * 2.5
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        # If all retries failed
        raise HuggingFaceServiceError(
            f"Hugging Face Inference API temporary error: {str(last_error)}. Please click Generate again.",
            status_code=503
        )

