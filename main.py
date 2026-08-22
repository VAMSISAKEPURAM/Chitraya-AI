import logging
import os
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.utils.config import settings
from backend.agent.image_agent import image_agent
from backend.services.huggingface_service import HuggingFaceServiceError

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api_server")

app = FastAPI(
    title="Chitraya AI - FLUX.1 Schnell Generator API",
    description="API for AI Image Generation using LangChain, Groq LLM, and FLUX.1 Schnell on Hugging Face",
    version="1.0.0"
)

# Enable CORS for local testing and iframe embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Natural language image prompt")

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend status and token configurations."""
    return {
        "status": "online",
        "hf_configured": settings.is_hf_configured(),
        "groq_configured": settings.is_groq_configured(),
        "image_model": settings.IMAGE_MODEL,
        "groq_model": settings.GROQ_MODEL
    }

@app.post("/api/generate-image")
async def generate_image(request: GenerateImageRequest):
    """
    Generate an AI image based on user prompt.
    Uses LangChain + Groq LLM to optimize the prompt,
    and FLUX.1 Schnell on Hugging Face to generate the image.
    """
    clean_prompt = request.prompt.strip()
    if not clean_prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt cannot be empty or blank."
        )

    try:
        result = image_agent.generate(clean_prompt)
        return result

    except HuggingFaceServiceError as hf_err:
        logger.error(f"HuggingFace Service Error: {hf_err.message}")
        raise HTTPException(
            status_code=hf_err.status_code,
            detail=hf_err.message
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as err:
        logger.exception("Unexpected error during image generation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(err)}"
        )

# Mount static frontend files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_index():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return JSONResponse({"message": "Chitraya AI API is running. Static UI not found."})
else:
    @app.get("/")
    async def serve_root():
        return JSONResponse({
            "message": "Chitraya AI - LangChain + FLUX.1 Schnell Image Generator API",
            "status": "online",
            "docs": "/docs"
        })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)
