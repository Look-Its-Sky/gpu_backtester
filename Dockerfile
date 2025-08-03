FROM nvcr.io/nvidia/rapidsai/base:25.06-cuda12.8-py3.13

WORKDIR /app

RUN conda install uv 
COPY requirements.txt .
RUN uv pip install -r requirements.txt

CMD ["uv", "run", "main.py"]
