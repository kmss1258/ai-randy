FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -f cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends libcudnn9-cuda-12 libcublaslt11 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir opencv-fixer==0.2.5
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
RUN pip install --no-cache-dir -r requirements.txt

# Ensure we end up with the GPU onnxruntime module (rembg pulls CPU onnxruntime).
RUN pip uninstall -y onnxruntime || true
RUN pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.20.0

# Ensure we end up with headless OpenCV only.
RUN pip uninstall -y opencv-python opencv-python-headless || true
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless==4.8.0.74

# opencv-fixer may pull in NumPy 2.x; force NumPy 1.x for binary wheels.
RUN pip install --no-cache-dir --force-reinstall --no-deps numpy==1.26.4

EXPOSE 19167

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "19167"]
