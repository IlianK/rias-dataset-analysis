# Latest Tensorflow GPU image
FROM tensorflow/tensorflow:latest-gpu

# Environment variables
ENV PYTHONPATH /workspace/ilian/src
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV TF_GPU_ALLOCATOR=cuda_malloc_async

# Set working directory
WORKDIR /app

# OpenCV
RUN pip install --upgrade pip 

# Install required packages
RUN python3 -m pip install -U pip
COPY docker/requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt