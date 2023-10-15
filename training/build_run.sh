#!/bin/bash

set -e

docker build -t training_container .

docker run \
    --gpus all \
    --shm-size=2G \
    -v $(pwd):/home/user/ \
    -v $(pwd)/../data:/home/user/data \
    -it \
    training_container:latest \
    bash
