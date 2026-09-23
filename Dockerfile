FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first: this API only does CPU inference, and the
# default PyPI wheel pulls in several GB of CUDA libraries we'd never use
# in the container.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY configs/api.yaml configs/api.yaml
COPY data/processed/eda_stats.json data/processed/eda_stats.json
COPY data/processed/transformer_full_checkpoints/final data/processed/transformer_full_checkpoints/final

# The model is loaded from the local path above, not downloaded from the
# Hub -- disable Hugging Face's hub access entirely so the container never
# makes an unexpected outbound network call at runtime.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "arxiv_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]