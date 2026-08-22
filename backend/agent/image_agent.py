import logging
import os
from datetime import datetime
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.utils.config import settings
from backend.agent.prompts import PROMPT_ENHANCER_SYSTEM_PROMPT
from backend.agent.tools import flux_image_generation_tool
from backend.services.huggingface_service import HuggingFaceServiceError

logger = logging.getLogger("image_agent")

class LangChainImageAgent:
    """
    LangChain Agent for Image Generation.
    1. Uses Groq LLM to understand and enhance the user's natural language prompt.
    2. Calls the FLUX.1 Schnell image generation tool on Hugging Face.
    """

    def __init__(self):
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", PROMPT_ENHANCER_SYSTEM_PROMPT),
            ("user", "{user_prompt}")
        ])

    def enhance_prompt_with_groq(self, user_prompt: str) -> str:
        """
        Uses Groq LLM via LangChain to understand intent and enrich the prompt.
        Falls back to fallback prompt expansion if Groq is unconfigured or unavailable.
        """
        if not settings.is_groq_configured():
            logger.info("Groq API Key not configured. Using fallback prompt expansion.")
            return self._fallback_prompt_enhancer(user_prompt)

        try:
            from langchain_groq import ChatGroq
            logger.info(f"Enhancing prompt using Groq LLM ({settings.GROQ_MODEL})...")
            
            # Set env var as fallback authentication method
            os.environ.setdefault("GROQ_API_KEY", settings.GROQ_API_KEY)
            
            llm = ChatGroq(
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0.7,
                max_tokens=250
            )

            chain = self.prompt_template | llm | StrOutputParser()
            enhanced = chain.invoke({"user_prompt": user_prompt}).strip()
            
            # Clean up potential wrapping quotes
            if (enhanced.startswith('"') and enhanced.endswith('"')) or (enhanced.startswith("'") and enhanced.endswith("'")):
                enhanced = enhanced[1:-1].strip()
                
            return enhanced if enhanced else user_prompt

        except Exception as e:
            logger.warning(f"Groq LLM prompt enhancement failed ({e}). Falling back to simple expansion.")
            return self._fallback_prompt_enhancer(user_prompt)

    def _fallback_prompt_enhancer(self, prompt: str) -> str:
        """Rule-based fallback prompt enhancer when Groq LLM is not active."""
        clean = prompt.strip()
        if len(clean.split()) < 5:
            return f"{clean}, cinematic lighting, photorealistic details, high quality, highly detailed composition."
        return clean

    def generate(self, user_prompt: str) -> Dict[str, Any]:
        """
        Main execution flow:
        Raw User Prompt -> LangChain Groq Prompt Optimization -> FLUX.1 Schnell Tool -> Final Result Payload
        """
        clean_prompt = user_prompt.strip()
        if not clean_prompt:
            raise ValueError("Prompt cannot be empty.")

        logger.info(f"Processing new image generation request for prompt: '{clean_prompt}'")
        
        # Step 1: Groq LLM Prompt Enhancement
        enhanced_prompt = self.enhance_prompt_with_groq(clean_prompt)
        logger.info(f"Enhanced prompt: '{enhanced_prompt}'")

        # Step 2: Invoke Image Generation Tool
        tool_result = flux_image_generation_tool.invoke({"prompt": enhanced_prompt})

        # Step 3: Return Response
        return {
            "success": True,
            "original_prompt": clean_prompt,
            "enhanced_prompt": enhanced_prompt,
            "image": tool_result["image"],
            "model": settings.IMAGE_MODEL,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

# Global agent instance
image_agent = LangChainImageAgent()
