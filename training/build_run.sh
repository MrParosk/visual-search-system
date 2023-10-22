#!/bin/bash

set -e

docker build -t training_container .

docker run \
    --gpus all \
    --shm-size=2G \
    -v $(pwd):/home/user/workdir/ \
    -v $(pwd)/../data:/home/user/data \
    -v $(pwd)/../artifact:/home/user/artifact \
    -it \
    training_container:latest \
    bash
