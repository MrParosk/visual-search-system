#!/bin/bash

set -e

docker build -t nearest_neighbor_index_container .

docker run \
    --gpus all \
    --shm-size=2G \
    -v $(pwd):/home/user/ \
    -v $(pwd)/../data:/home/user/data \
    -v $(pwd)/../artifact:/home/user/artifact \
    -it \
    nearest_neighbor_index_container:latest \
    bash
