#!/bin/bash

set -e

kaggle datasets download -d imbikramsaha/caltech-101
mv caltech-101.zip data/
unzip data/caltech-101.zip -d data
