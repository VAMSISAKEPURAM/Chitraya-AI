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

# 🎨 Chitraya AI — LangChain + Groq + FLUX.1 Schnell

An AI image generation application powered by **LangChain**, **Groq LLM** (for visual prompt enhancement & intent understanding), and **FLUX.1 Schnell** via Hugging Face Inference API.

## Features
- 🧠 **LangChain Prompt Engineering Agent**: Converts simple descriptions into photographic, high-detail visual prompts tailored for FLUX.1 Schnell.
- ⚡ **FLUX.1 Schnell Generation**: Ultra-fast text-to-image synthesis.
- 🎨 **Modern Interface**: Dark mode, interactive example prompts, original vs. enhanced prompt inspection, and instant downloads.
- 🛡️ **ZeroGPU & CPU Compatible**: Runs seamlessly on free Hugging Face Spaces (CPU or ZeroGPU).

## Setup & Secrets
In your Hugging Face Space **Settings → Variables and Secrets**, configure:
- `HF_TOKEN`: Your Hugging Face user access token.
- `GROQ_API_KEY`: Your Groq Cloud API key (optional — falls back to rule-based enhancement if not provided).

## Local Development
```bash
git clone https://github.com/VAMSISAKEPURAM/Chitraya-AI.git
cd Chitraya-AI
pip install -r requirements.txt
python app.py
```
Open `http://localhost:7860` in your browser.
