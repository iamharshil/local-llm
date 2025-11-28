FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers==4.41.2 \
    datasets==2.19.0 \
    peft==0.10.0 \
    accelerate \
    sentencepiece \
    bitsandbytes \
    fastapi \
    uvicorn \
    huggingface_hub \
    llama-cpp-python

WORKDIR /app
COPY . .

EXPOSE 8000

CMD ["python", "api/server.py"]
