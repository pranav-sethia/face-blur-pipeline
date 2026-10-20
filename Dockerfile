FROM python:3.11-slim

# System libs required by opencv-python at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch's default PyPI wheel bundles the full CUDA runtime (several GB) even though this pipeline
# only ever runs torch on CPU -- the CPU-only wheel index avoids pulling that in unnecessarily.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY pipeline/ ./pipeline/
COPY models/ ./models/

# Bake the SCRFD weight into the fixed path insightface's loader expects, so no manual setup
# step is needed inside the container.
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    cp models/det_10g.onnx /root/.insightface/models/buffalo_l/det_10g.onnx

ENTRYPOINT ["python", "pipeline/blur_faces.py"]
