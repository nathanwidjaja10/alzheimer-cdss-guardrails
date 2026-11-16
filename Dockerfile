# Runtime-only dependency image for the backend (see requirements-runtime.txt). Source, data and checkpoints
# are bind-mounted at /app by docker-compose.yml (OASIS-3/ADNI are DUA-restricted and
# multi-GB: they must never be baked into an image).
FROM python:3.10-slim

# torch/sklearn need OpenMP; curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels by default (PyPI's Linux x86 torch pulls ~GBs of CUDA libs). For an NVIDIA GPU:
#   docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

WORKDIR /tmp/deps
COPY requirements-runtime.txt .
# torch/torchvision first from the chosen index (PyPI as extra index for their deps; the +cpu/+cuXXX
# local version outranks the plain PyPI one); the rest from PyPI.
RUN grep -E '^(torch|torchvision)==' requirements-runtime.txt > torch.txt \
 && grep -v -E '^(torch|torchvision)==' requirements-runtime.txt > rest.txt \
 && pip install --no-cache-dir -r torch.txt --index-url ${TORCH_INDEX} --extra-index-url https://pypi.org/simple \
 && pip install --no-cache-dir -r rest.txt

WORKDIR /app
ENV PYTHONUNBUFFERED=1 BASE_DIR=/app PYTHONPATH=/app
