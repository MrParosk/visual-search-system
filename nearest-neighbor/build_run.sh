#!/bin/bash

set -e

docker build -t nearest_neighbor_container .

docker run \
    -v $(pwd):/home/user/ \
    -v $(pwd)/../data:/home/user/data \
    -v $(pwd)/../artifact:/home/user/artifact \
    -it \
    nearest_neighbor_container:latest \
    bash
