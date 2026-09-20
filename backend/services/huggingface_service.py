import base64
import io
import time
import logging
import requests
from PIL import Image
from typing import Tuple

from backend.utils.config import settings

logger = logging.getLogger("huggingface_service")

# Candidate fallback models in priority order if primary is busy/overloaded
FALLBACK_MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
    "runwayml/stable-diffusion-v1-5"
]

class HuggingFaceServiceError(Exception):
    """Custom exception for Hugging Face Inference API operations."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class HuggingFaceService:
    """Service to interact with Hugging Face Inference API with automatic retry and model fallback."""

    @staticmethod
    def generate_image(prompt: str) -> Tuple[str, Image.Image, str]:
        """
        Generates an image from a prompt using FLUX.1 Schnell on Hugging Face.
        Automatically retries and falls back to secondary models if primary is overloaded.
        Returns a tuple of (base64_data_uri, PIL_Image, model_used).
        """
        if not settings.is_hf_configured():
            raise HuggingFaceServiceError(
                "Hugging Face API Token (HF_TOKEN) is not configured. Please add your token in .env or Space Settings -> Secrets.",
                status_code=401
            )

        token = settings.HF_TOKEN
        primary_model = settings.IMAGE_MODEL

        # Build prioritized list of models to try
        models_to_try = [primary_model]
        for m in FALLBACK_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

        last_error = None

        for model in models_to_try:
            # Try each model with up to 2 attempts
            for attempt in range(1, 3):
                try:
                    logger.info(f"Attempting image generation with model '{model}' (Attempt {attempt}/2)...")
                    data_uri, image = HuggingFaceService._try_inference_client(prompt, model, token)
                    logger.info(f"Successfully generated image using '{model}'!")
                    return data_uri, image, model

                except Exception as client_err:
                    err_str = str(client_err).lower()
                    logger.warning(f"InferenceClient failed for '{model}' (Attempt {attempt}): {client_err}")

                    # Try direct HTTP fallback
                    try:
                        data_uri, image = HuggingFaceService._generate_via_http(prompt, model, token)
                        logger.info(f"Successfully generated image via HTTP using '{model}'!")
                        return data_uri, image, model
                    except Exception as http_err:
                        last_error = http_err
                        logger.warning(f"HTTP fallback also failed for '{model}': {http_err}")

                    # If temporary overload/busy error, wait briefly before retrying
                    if attempt < 2 and any(k in err_str for k in ["overloaded", "503", "429", "timeout", "busy", "loading"]):
                        time.sleep(2.0)

        # If all models and retries failed, raise descriptive error
        error_msg = str(last_error) if last_error else "All Hugging Face model endpoints are temporarily unavailable."
        raise HuggingFaceServiceError(
            f"Image generation failed across models: {error_msg}. Please retry in a few seconds.",
            status_code=503
        )

    @staticmethod
    def _try_inference_client(prompt: str, model: str, token: str) -> Tuple[str, Image.Image]:
        from huggingface_hub import InferenceClient
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
        return data_uri, image

    @staticmethod
    def _generate_via_http(prompt: str, model: str, token: str) -> Tuple[str, Image.Image]:
        """Fallback direct HTTP request implementation to Hugging Face Serverless Inference API."""
        api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
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

        response = requests.post(api_url, headers=headers, json=payload, timeout=45)

        if response.status_code == 401:
            raise HuggingFaceServiceError(
                "Invalid or unauthorized Hugging Face token. Please check HF_TOKEN in Space Settings -> Secrets.",
                status_code=401
            )
        elif response.status_code == 429:
            raise HuggingFaceServiceError("Hugging Face API rate limit reached.", status_code=429)
        elif response.status_code == 503:
            raise HuggingFaceServiceError(f"Model '{model}' is currently loading or overloaded on Hugging Face.", status_code=503)
        elif response.status_code != 200:
            raise HuggingFaceServiceError(f"Hugging Face API error ({response.status_code}): {response.text[:200]}", status_code=response.status_code)

        image = Image.open(io.BytesIO(response.content))
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{img_str}"
        return data_uri, image
