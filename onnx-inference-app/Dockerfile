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

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir 'numpy<2.0.0' && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY onnxmodels/yolo.onnx onnxmodels/vgglandmark.onnx onnxmodels/category.onnx /app/onnxmodels/
COPY inference_onnx.py models.py processing.py labels.py /app/

# Final minimal image
FROM python:3.9.19-slim
WORKDIR /app

# Copy everything from builder
COPY --from=builder /app /app

EXPOSE 5000
CMD ["python", "inference_onnx.py"]
