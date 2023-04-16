#!/bin/bash

if [ "$1" ]; then
    MODEL_REPO_NAME=$1

    # default to current directory if unspecified
    DOWNLOAD_PATH=${2:-.}     

    mkdir -p "$DOWNLOAD_PATH/$MODEL_REPO_NAME"

    cd "$DOWNLOAD_PATH/$MODEL_REPO_NAME"

    git lfs install

    git clone -b main --single-branch --depth 1 "https://huggingface.co/$MODEL_REPO_NAME" .
else
    echo $'Usage: ./download_model.sh <model_repo_name> [download_path]\n\nPlease specify the required model_repo_name or optional download_path (default to current directory).'
fi

