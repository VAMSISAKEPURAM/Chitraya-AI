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

<!-- Trigger build update: Gradio 5.16.0 -->


# Chitraya AI - LangChain + FLUX.1 Schnell Image Generator

An AI image generation web application powered by **LangChain**, **Groq LLM** (for prompt understanding & optimization), **FLUX.1 Schnell** via Hugging Face Inference API, and a FastAPI backend with a modern glassmorphism frontend.

## Features
- **Prompt Engineering Agent**: Uses Groq LLM to enrich raw user prompts into detailed photographic descriptive prompts tailored for FLUX.1 Schnell.
- **FLUX.1 Schnell Generation**: High-speed, high-quality text-to-image synthesis.
- **Modern UI**: Dark mode, glassmorphism aesthetics, prompt comparison, multi-stage loading indicator, image preview modal, instant download, and generation history.

## Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/VAMSISAKEPURAM/Chitraya-AI.git
   cd Chitraya-AI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` file:
   ```env
   HF_TOKEN=your_huggingface_token
   GROQ_API_KEY=your_groq_api_key
   IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

4. Run the app:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:7860` in your browser.

