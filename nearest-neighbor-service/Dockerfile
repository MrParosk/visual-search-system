FROM python:3.11-bookworm

RUN apt-get update && apt-get upgrade -y

RUN pip install torch==2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install faiss-cpu==1.7.4 fastapi==0.104.0 python-multipart==0.0.6 uvicorn==0.23.2

WORKDIR /home/user/workdir/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]