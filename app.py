import os
import io
import base64
import logging
import gradio as gr
from PIL import Image

from backend.utils.config import settings
from backend.agent.image_agent import image_agent
from backend.services.huggingface_service import HuggingFaceServiceError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("chitraya_app")

# ─── Core Generation Function ─────────────────────────────────────────────────
# NOTE: No @spaces.GPU decorator needed — this app calls the remote HF Inference
# API and does NOT run any model locally. ZeroGPU allocation is unnecessary.
def generate_image(prompt: str):
    """
    Main generation pipeline:
    User Prompt → Groq LLM Enhancement → FLUX.1 Schnell (remote API) → Image
    """
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter or select a prompt before clicking Generate.")

    clean_prompt = prompt.strip()
    logger.info(f"Generate request received: '{clean_prompt[:80]}...'")

    try:
        result = image_agent.generate(clean_prompt)
        logger.info("Image generation completed successfully.")
    except HuggingFaceServiceError as e:
        logger.error(f"HuggingFace Service Error: {e.message}")
        raise gr.Error(f"Hugging Face API Error: {e.message}")
    except Exception as e:
        logger.error(f"Generation Error: {str(e)}", exc_info=True)
        raise gr.Error(f"Generation Error: {str(e)}")

    # Parse image data
    data_uri = result.get("image", "")
    if isinstance(data_uri, str) and data_uri.startswith("data:image"):
        header, b64data = data_uri.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        image = Image.open(io.BytesIO(img_bytes))
    elif isinstance(data_uri, Image.Image):
        image = data_uri
    else:
        logger.error(f"Unexpected image data type: {type(data_uri)}")
        raise gr.Error("Failed to parse image from generator output.")

    model_info = result.get("model", settings.IMAGE_MODEL)
    return (
        image,
        result.get("enhanced_prompt", clean_prompt),
        result.get("original_prompt", clean_prompt),
        f"Generated via: {model_info}"
    )

def get_status():
    hf_ok = settings.is_hf_configured()
    groq_ok = settings.is_groq_configured()
    hf_icon = "🟢 Ready" if hf_ok else "🔴 Missing Secret (HF_TOKEN)"
    groq_icon = "🟢 Active (LLM prompt expander)" if groq_ok else "🟡 Rule-based mode"
    return f"**Hugging Face Inference:** {hf_icon} &nbsp;|&nbsp; **Groq Agent:** {groq_icon} &nbsp;|&nbsp; **Model:** `{settings.IMAGE_MODEL}`"

# ─── Preset Prompts ──────────────────────────────────────────────────────────
PROMPT_1 = "A realistic Indian farmer working in a smart agricultural field with golden hour lighting"
PROMPT_2 = "A futuristic cyberpunk city at sunset with neon reflections and flying autonomous vehicles"
PROMPT_3 = "A luxury black sports car in a cinematic dark studio with dramatic neon rim lighting"
PROMPT_4 = "A cute astronaut cat walking on Mars with Earth visible in the star-filled sky, digital painting"

theme = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_700",
    input_background_fill="*neutral_800",
    button_primary_background_fill="linear-gradient(135deg, #7c3aed, #a855f7)",
    button_primary_background_fill_hover="linear-gradient(135deg, #6d28d9, #9333ea)",
    button_primary_text_color="white",
)

css = """
#title { text-align: center; margin-bottom: 8px; }
#title h1 { 
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 2px;
}
#title p { color: #94a3b8; font-size: 0.95rem; }
#status-box { border-radius: 8px; padding: 8px 14px; background: #1e293b; font-size: 0.88rem; }
#gen-btn { height: 50px; font-size: 1.1rem; font-weight: 700; margin-top: 10px; }
#output-image { border-radius: 12px; }
.quick-chip { margin: 2px 0; }
"""

import inspect

# Dynamically route theme and css to Blocks or launch depending on Gradio version (Gradio 5 vs 6)
blocks_params = inspect.signature(gr.Blocks.__init__).parameters
launch_params = inspect.signature(gr.Blocks.launch).parameters

blocks_kwargs = {"title": "Chitraya AI - FLUX.1 Schnell Generator"}
launch_kwargs = {
    "server_name": "0.0.0.0",
    "server_port": int(os.getenv("PORT", 7860)),
}

if "theme" in blocks_params:
    blocks_kwargs["theme"] = theme
elif "theme" in launch_params:
    launch_kwargs["theme"] = theme

if "css" in blocks_params:
    blocks_kwargs["css"] = css
elif "css" in launch_params:
    launch_kwargs["css"] = css

with gr.Blocks(**blocks_kwargs) as demo:

    # Header & Status
    with gr.Column(elem_id="title"):
        gr.HTML("<h1>🎨 Chitraya AI</h1><p>LangChain Agent + Groq Prompt Optimization · FLUX.1 Schnell · ZeroGPU</p>")

    status_md = gr.Markdown(get_status(), elem_id="status-box")

    with gr.Row():
        # Left side: Prompt Controls
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="✏️ Describe the image you want to create",
                placeholder="Type your prompt here or click any example below...",
                lines=4,
                max_lines=6,
                elem_id="prompt-input"
            )

            gr.Markdown("**💡 Quick Example Prompts (click to set):**")
            with gr.Column():
                btn1 = gr.Button(f"🌾 {PROMPT_1}", size="sm", elem_classes=["quick-chip"])
                btn2 = gr.Button(f"🌃 {PROMPT_2}", size="sm", elem_classes=["quick-chip"])
                btn3 = gr.Button(f"🏎️ {PROMPT_3}", size="sm", elem_classes=["quick-chip"])
                btn4 = gr.Button(f"🚀 {PROMPT_4}", size="sm", elem_classes=["quick-chip"])

            generate_btn = gr.Button(
                "✨ Generate Image",
                variant="primary",
                elem_id="gen-btn"
            )

            with gr.Accordion("📋 Prompt Inspection & LangChain Agent Output", open=False):
                model_used_box = gr.Textbox(
                    label="⚡ Model Engine Used",
                    interactive=False,
                    lines=1,
                )
                original_prompt_box = gr.Textbox(
                    label="Your Original Prompt",
                    interactive=False,
                    lines=2,
                )
                enhanced_prompt_box = gr.Textbox(
                    label="🤖 LangChain + Groq Enhanced Prompt (sent to Model)",
                    interactive=False,
                    lines=4,
                )

        # Right side: Generated Output
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="🖼️ Generated AI Artwork",
                type="pil",
                elem_id="output-image",
                height=500,
            )

    # Wire up Quick Example buttons to set the prompt text immediately
    btn1.click(fn=lambda: PROMPT_1, inputs=[], outputs=[prompt_input])
    btn2.click(fn=lambda: PROMPT_2, inputs=[], outputs=[prompt_input])
    btn3.click(fn=lambda: PROMPT_3, inputs=[], outputs=[prompt_input])
    btn4.click(fn=lambda: PROMPT_4, inputs=[], outputs=[prompt_input])

    # Wire up Generate button & Enter key
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input],
        outputs=[output_image, enhanced_prompt_box, original_prompt_box, model_used_box],
        api_name="generate",
    )

    prompt_input.submit(
        fn=generate_image,
        inputs=[prompt_input],
        outputs=[output_image, enhanced_prompt_box, original_prompt_box, model_used_box],
    )

    gr.Markdown(
        "---\n*Powered by LangChain · Groq openai/gpt-oss-120b · FLUX.1 Schnell via Hugging Face Inference API · Gradio*",
    )

if __name__ == "__main__":
    demo.launch(**launch_kwargs)


