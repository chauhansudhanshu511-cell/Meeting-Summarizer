# Deploys the Streamlit app on Hugging Face Spaces (Docker SDK) or any Docker host.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN python -m spacy download en_core_web_sm

COPY --chown=user . .

# Bake the AI models into the image so the first request is not slow
RUN python download_models.py

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
