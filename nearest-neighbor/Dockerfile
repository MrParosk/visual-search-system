FROM python:3.11-bookworm

RUN apt-get update && apt-get upgrade -y

RUN pip install torch==2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install faiss-cpu==1.7.4

WORKDIR /home/user/
