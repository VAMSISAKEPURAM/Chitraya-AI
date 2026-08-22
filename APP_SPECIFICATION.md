# 🎨 Chitraya AI — Complete System Blueprint & Architecture Reference

> **Document Type:** Master Reference Specification Document  
> **Target System:** Chitraya AI (AI Image Generation Web Application)  
> **Key Technologies:** Python 3.10+, FastAPI, Gradio 5+, LangChain Core / Groq, FLUX.1 Schnell, Hugging Face Inference API / ZeroGPU  
> **Repository:** `VAMSISAKEPURAM/Chitraya-AI`  
> **Date:** August 2026

---

## 1. System Overview & Core Objectives

**Chitraya AI** is an intelligent text-to-image synthesis application that turns natural language user prompts into photographic, high-detail AI artwork.

### Core Workflow:
1. **User Request**: User inputs a short or natural language description (e.g., *"A realistic Indian farmer working in a smart agricultural field"*).
2. **LangChain + Groq Agent**: An AI Prompt Engineering agent intercepts the raw prompt and expands it into an optimized, descriptive visual prompt tailored specifically for **FLUX.1 Schnell** (lighting, texture, composition, atmosphere, camera lens characteristics). If Groq is unavailable, it gracefully uses a rule-based enhancer.
3. **FLUX.1 Schnell Image Generator**: The enriched prompt is dispatched to the Hugging Face Inference API (or local `InferenceClient`) using the `black-forest-labs/FLUX.1-schnell` model.
4. **Interactive UI Delivery**: The output image, along with both original and agent-enhanced prompts, is returned and rendered in a modern glassmorphism web interface and Gradio interface with instant download, full-screen preview, and history logging.

---

## 2. Architecture & Directory Layout

The application is structured in a modular, clean Python architecture:

```
text-to-image/
├── APP_SPECIFICATION.md          # Complete master architectural & reference blueprint
├── README.md                     # Hugging Face Space metadata & documentation
├── requirements.txt              # Pinned, conflict-free Python dependencies
├── .env.example                  # Environment variable reference
├── .env                          # Local credentials (git-ignored)
├── .gitignore                    # Ignored directories & cache
├── app.py                        # Unified Gradio + FastAPI deployment entry point
├── main.py                       # Core FastAPI REST API server & static file host
├── backend/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── image_agent.py        # LangChain agent orchestrator & Groq prompt enhancer
│   │   ├── prompts.py            # Visual Art Director system prompt templates
│   │   └── tools.py              # FLUX.1 Schnell tool wrapper
│   ├── services/
│   │   ├── __init__.py
│   │   └── huggingface_service.py # HF Inference Client & fallback HTTP API caller
│   └── utils/
│       ├── __init__.py
│       └── config.py             # Dynamic environment & secrets configuration loader
└── static/
    ├── index.html                # Modern glassmorphism web frontend
    ├── style.css                 # Dark theme, glassmorphism, responsive styles
    └── app.js                    # Dynamic frontend script (fetch API, history, modals)
```

---

## 3. Component Deep Dive & Logic Specifications

### A. Configuration Management (`backend/utils/config.py`)
- Reads environment variables with dynamic property getters to prevent stale cached imports.
- Supports all common secret names for Hugging Face and Groq:
  - `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_API_TOKEN`
  - `GROQ_API_KEY`, `GROQ_TOKEN`
- Default Models:
  - `IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"`
  - `GROQ_MODEL = "llama-3.3-70b-versatile"`

### B. Prompt Engineering System (`backend/agent/prompts.py`)
- **System Prompt**:
  - Acts as an expert AI Visual Art Director.
  - Expands prompts with depth, lighting (golden hour, volumetric, dramatic rim light), attire, environment, camera angles, and textures.
  - Keeps natural fluent English paragraphs without buzzword stuffing (`masterpiece`, `8k`, etc.).
  - Returns strictly the refined prompt text with no conversational filler.

### C. LangChain Agent Orchestrator (`backend/agent/image_agent.py`)
- **Class**: `LangChainImageAgent`
- **Methods**:
  - `enhance_prompt_with_groq(prompt: str) -> str`: Uses `ChatGroq(model=..., api_key=...)` with fallback to rule-based expansion if Groq key is absent or API limit reached.
  - `generate(user_prompt: str) -> dict`: Orchestrates enhancement → Tool execution → formatted payload output:
    ```json
    {
      "success": true,
      "original_prompt": "...",
      "enhanced_prompt": "...",
      "image": "data:image/png;base64,...",
      "model": "black-forest-labs/FLUX.1-schnell",
      "timestamp": "2026-08-22T..."
    }
    ```

### D. Hugging Face Image Synthesis Service (`backend/services/huggingface_service.py`)
- **Class**: `HuggingFaceService`
- **Method**: `generate_image(prompt: str) -> (data_uri: str, pil_image: Image)`
- **Strategy**:
  1. Primary: `huggingface_hub.InferenceClient(token=token).text_to_image(prompt=prompt, model=model)`
  2. Fallback: Direct HTTPS POST to `https://api-inference.huggingface.co/models/{model}` with `guidance_scale: 0.0` and `num_inference_steps: 4`.
  3. Output conversion: Encodes PIL image to Base64 data URI format (`data:image/png;base64,...`).

### E. FastAPI REST API Layer (`main.py`)
- **Endpoints**:
  - `GET /api/health`: Returns service status, token configuration flags, and active model names.
  - `POST /api/generate-image`: Accepts `{"prompt": "string"}`, executes agent pipeline, and returns JSON payload.
  - `GET /`: Serves `static/index.html`.
  - Static Files Mount: Mounts `/static` directory for CSS, JS, and media assets.
  - CORS Middleware: Allows all origins for local testing and iframe embedding.

### F. Gradio & ZeroGPU Integration Layer (`app.py`)
- **Gradio 5+ Compatible**:
  - Provides a complete, responsive dark-themed Gradio Blocks application.
  - Contains preset prompt buttons (🌾 Smart Agriculture, 🌃 Cyberpunk City, 🏎️ Luxury Studio Car, 🚀 Astronaut Cat) that directly populate the input box on click.
  - Top-level `@spaces.GPU(duration=60)` decorator for Hugging Face ZeroGPU compatibility.
  - No nested `@spaces.GPU` calls to prevent process deadlocks.
  - Can run standalone on port 7860 or be mounted.

### G. Glassmorphism Web Frontend (`static/`)
- **Visual Design System**:
  - Deep dark theme (`#0a0f1d`), glowing background globes, glassmorphism cards with backdrop blur.
  - Multi-stage animated loading stepper (Analyzing intent → Optimizing prompt → Synthesizing FLUX.1 image → Rendering).
  - Side-by-side prompt comparison (User Original vs. LangChain Agent Enhanced) with copy-to-clipboard button.
  - Fullscreen image preview modal and 1-click image download with automatic filename slugification.
  - `localStorage` generation history grid with thumbnail preview and click-to-restore.

---

## 4. Dependencies & Deployment Setup

### `requirements.txt`
```
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
langchain-groq>=0.2.0
huggingface_hub>=0.25.0
fastapi>=0.115.0
uvicorn>=0.30.0
python-dotenv>=1.0.0
Pillow>=10.0.0
requests>=2.31.0
pydantic>=2.0.0
gradio>=5.16.0
python-multipart>=0.0.9
httpx>=0.27.0
spaces
```

### `README.md` Frontmatter (for Hugging Face Spaces):
```yaml
---
title: Chitraya AI - FLUX.1 Image Generator
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
---
```

---

## 5. Deployment Checklist & Zero-Error Guarantees

1. **ZeroGPU Safety**: Only **one** `@spaces.GPU` decorator at the topmost generator function in `app.py`.
2. **Graceful Token Fallbacks**: If `GROQ_API_KEY` is not provided, agent automatically uses rule-based prompt expansion without crashing.
3. **Clear Error Notifications**: All errors (rate limits, invalid tokens, model cold start) return structured user-friendly messages.
4. **Dual Interface Support**: Both Gradio interface and FastAPI Custom Glassmorphism interface are fully operational.
