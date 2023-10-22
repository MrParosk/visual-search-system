#!/bin/bash

set -e

docker build -t nearest_neighbor_service_container .

docker run \
    -v $(pwd):/home/user/workdir/ \
    -v $(pwd)/../data:/home/user/data \
    -v $(pwd)/../artifact:/home/user/artifact \
    -p 8080:8080 \
    -it \
    nearest_neighbor_service_container:latest \
