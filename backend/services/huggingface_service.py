import base64
import io
import logging
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
    """Service to interact with Hugging Face Inference API for FLUX.1 Schnell image generation."""

    @staticmethod
    def generate_image(prompt: str) -> Tuple[str, Image.Image]:
        """
        Generates an image from a prompt using FLUX.1 Schnell on Hugging Face.
        Returns a tuple of (base64_data_uri, PIL_Image).
        """
        if not settings.is_hf_configured():
            raise HuggingFaceServiceError(
                "Hugging Face API Token (HF_TOKEN) is not configured. Please add your token in Space Settings -> Secrets.",
                status_code=401
            )

        token = settings.HF_TOKEN
        model = settings.IMAGE_MODEL

        # Attempt generation using huggingface_hub InferenceClient first
        try:
            from huggingface_hub import InferenceClient
            logger.info(f"Sending prompt to Hugging Face InferenceClient (Model: {model})...")
            client = InferenceClient(token=token)
            
            # FLUX.1 Schnell text-to-image call
            image = client.text_to_image(prompt=prompt, model=model)
            
            # Convert PIL Image to Base64 data URI
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{img_str}"
            
            return data_uri, image

        except Exception as client_err:
            logger.warning(f"InferenceClient call failed ({client_err}). Attempting fallback HTTP API request...")
            return HuggingFaceService._generate_via_http(prompt, model, token)

    @staticmethod
    def _generate_via_http(prompt: str, model: str, token: str) -> Tuple[str, Image.Image]:
        """Fallback direct HTTP request implementation to Hugging Face Serverless Inference API."""
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale": 0.0,
                "num_inference_steps": 4
            }
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        except requests.exceptions.Timeout:
            raise HuggingFaceServiceError(
                "Request to Hugging Face timed out. Please try again.",
                status_code=504
            )
        except requests.exceptions.RequestException as req_err:
            raise HuggingFaceServiceError(
                f"Network error connecting to Hugging Face: {str(req_err)}",
                status_code=502
            )

        if response.status_code == 401:
            raise HuggingFaceServiceError(
                "Invalid or unauthorized Hugging Face token. Please check HF_TOKEN under Space Settings -> Secrets.",
                status_code=401
            )
        elif response.status_code == 429:
            raise HuggingFaceServiceError(
                "Hugging Face API rate limit reached. Please wait a moment and try again.",
                status_code=429
            )
        elif response.status_code == 503:
            raise HuggingFaceServiceError(
                "FLUX.1 Schnell model is currently loading or busy on Hugging Face. Please try again in 10-20 seconds.",
                status_code=503
            )
        elif response.status_code != 200:
            error_msg = f"Hugging Face API error ({response.status_code}): {response.text[:250]}"
            raise HuggingFaceServiceError(error_msg, status_code=response.status_code)

        try:
            image = Image.open(io.BytesIO(response.content))
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{img_str}"
            return data_uri, image
        except Exception as img_err:
            raise HuggingFaceServiceError(
                f"Failed to process generated image output: {str(img_err)}",
                status_code=500
            )
