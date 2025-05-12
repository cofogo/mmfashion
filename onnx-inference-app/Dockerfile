FROM python:3.9-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
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

COPY requirements.txt /app

RUN pip install 'numpy<2.0.0'
RUN pip install -r requirements.txt

COPY yoloItem.onnx /app
COPY landmark.onnx /app

COPY inference_onnx.py /app
EXPOSE 5000
CMD ["python", "inference_onnx.py"]