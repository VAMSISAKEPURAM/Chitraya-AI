import logging
from langchain_core.tools import tool
from backend.services.huggingface_service import HuggingFaceService

logger = logging.getLogger("agent_tools")

@tool
def flux_image_generation_tool(prompt: str) -> dict:
    """
    Invokes the FLUX.1 Schnell image generation model on Hugging Face using the provided optimized prompt.
    Returns a dictionary containing the base64 image data URI string.
    """
    logger.info(f"Invoking FLUX.1 Schnell tool with prompt: {prompt[:80]}...")
    data_uri, _ = HuggingFaceService.generate_image(prompt)
    return {
        "status": "success",
        "image": data_uri,
        "prompt": prompt
    }
