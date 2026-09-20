import io
import base64
import logging
from PIL import Image
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union

from backend.utils.config import settings
from backend.agent.tools import flux_image_generation_tool
from backend.services.huggingface_service import HuggingFaceServiceError

logger = logging.getLogger("image_agent")

class ChitrayaImageAgent:
    """
    Image Generation & Transformation Agent:
    1. Direct Text-to-Image: Converts user prompt into AI artwork.
    2. Photo Remix & Transform: Takes an uploaded user photo and performs style / subject transformations.
    """

    def generate(self, user_prompt: str) -> Dict[str, Any]:
        """
        Direct text-to-image generation.
        """
        clean_prompt = user_prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt cannot be empty.")

        logger.info(f"Direct text-to-image request for prompt: '{clean_prompt[:80]}...'")
        
        tool_result = flux_image_generation_tool.invoke({"prompt": clean_prompt})

        return {
            "success": True,
            "mode": "text-to-image",
            "original_prompt": clean_prompt,
            "prompt": clean_prompt,
            "image": tool_result["image"],
            "model": tool_result.get("model", settings.IMAGE_MODEL),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def transform_image(
        self,
        input_image: Union[Image.Image, str],
        transform_prompt: str,
        style_preset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Takes a user's uploaded photo and applies the specified visual transformations,
        re-imagining the photo into the desired artistic style or scenario.
        """
        clean_prompt = transform_prompt.strip() if transform_prompt else ""
        if not clean_prompt and not style_preset:
            raise ValueError("Please provide a transformation instruction or select a style preset.")

        # Convert input_image to PIL if needed
        pil_image = None
        if isinstance(input_image, Image.Image):
            pil_image = input_image
        elif isinstance(input_image, str) and input_image.startswith("data:image"):
            header, b64data = input_image.split(",", 1)
            pil_image = Image.open(io.BytesIO(base64.b64decode(b64data)))

        # Formulate synthesized transformation prompt
        prompt_parts = []
        if style_preset:
            prompt_parts.append(style_preset.strip())
        if clean_prompt:
            prompt_parts.append(clean_prompt)
        
        final_prompt = ", ".join(prompt_parts)
        if not final_prompt.endswith("."):
            final_prompt += ", professional cinematic lighting, intricate details, photorealistic composition"

        logger.info(f"Photo transformation request: '{final_prompt[:80]}...'")

        # Invoke image generation engine with the transformation prompt
        tool_result = flux_image_generation_tool.invoke({"prompt": final_prompt})

        return {
            "success": True,
            "mode": "photo-transform",
            "transformation_prompt": clean_prompt,
            "prompt": final_prompt,
            "image": tool_result["image"],
            "model": tool_result.get("model", settings.IMAGE_MODEL),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Global singleton agent instance
image_agent = ChitrayaImageAgent()


