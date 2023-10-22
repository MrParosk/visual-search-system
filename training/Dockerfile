FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y gcc

WORKDIR /home/user/workdir/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
