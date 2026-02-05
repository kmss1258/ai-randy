FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=0

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir opencv-fixer==0.2.5
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"

EXPOSE 19167

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "19167"]
