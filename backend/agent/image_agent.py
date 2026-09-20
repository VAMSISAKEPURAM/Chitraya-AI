import logging
from datetime import datetime, timezone
from typing import Dict, Any

from backend.utils.config import settings
from backend.agent.tools import flux_image_generation_tool
from backend.services.huggingface_service import HuggingFaceServiceError

logger = logging.getLogger("image_agent")

class DirectImageAgent:
    """
    Direct Image Generation Agent:
    Takes the user's prompt exactly as provided and sends it directly
    to the Hugging Face text-to-image generation pipeline.
    """

    def generate(self, user_prompt: str) -> Dict[str, Any]:
        """
        Main execution flow:
        Raw User Prompt -> Direct Image Generation Tool -> Final Result Payload
        """
        clean_prompt = user_prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt cannot be empty.")

        logger.info(f"Direct image generation request for prompt: '{clean_prompt}'")
        
        # Directly invoke Image Generation Tool with raw user prompt
        tool_result = flux_image_generation_tool.invoke({"prompt": clean_prompt})

        # Return Response Payload
        return {
            "success": True,
            "original_prompt": clean_prompt,
            "prompt": clean_prompt,
            "image": tool_result["image"],
            "model": tool_result.get("model", settings.IMAGE_MODEL),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Global singleton agent instance
image_agent = DirectImageAgent()

