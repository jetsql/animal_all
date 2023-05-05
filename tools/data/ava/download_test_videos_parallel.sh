#!/usr/bin/env bash

set -e

DATA_DIR="/home/perry/Desktop/code/mmaction2/data/ava/test_video"
ANNO_DIR="/home/perry/Desktop/code/mmaction2/data/ava/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_test_v2.1.txt -P ${ANNO_DIR}

python download_test_videos_parallel.py ${ANNO_DIR}/ava_file_names_test_v2.1.txt ${DATA_DIR}
