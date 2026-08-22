FROM python:3.11-slim

# Create user with UID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application files
COPY --chown=user . .

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Run via app.py (includes Gradio mount + FastAPI custom UI)
# This is consistent with app_file: app.py in README.md frontmatter
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
