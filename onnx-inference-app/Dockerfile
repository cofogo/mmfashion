FROM python:3.9.19-slim AS builder
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxrender-dev \
    libxext6 \
    libfontconfig1 \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir 'numpy<2.0.0' && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY onnxmodels/* /app/onnxmodels/
COPY onnxmodels/attributeLayers/* /app/onnxmodels/attributeLayers/
COPY *.py /app/

# Final image
FROM python:3.9.19-slim
WORKDIR /app

# Reinstall system dependencies required at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxrender-dev \
    libxext6 \
    libfontconfig1 \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy app with installed packages
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /app /app

ENV YOLO_CONFIG_DIR=/app/.ultralytics

EXPOSE 5000
CMD ["python", "inference_onnx.py"]
