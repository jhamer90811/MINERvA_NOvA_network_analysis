#!/bin/bash

INPUT_PATH=$1
OUTPUT_PATH=$2
IMGS_PATH=$3
MODE=$4
GEN_PER_BATCH=$5

export i=1

while [($i + $GEN_PER_BATCH - 1) -lt 5000]
do
    end=${i} + ${GEN_PER_BATCH} - 1
    complex_job_submit.sh ${INPUT_PATH} ${OUTPUT_PATH} ${IMGS_PATH} ${i} ${end} ${MODE}
    i=${i} + ${GEN_PER_BATCH}
done

complex_job_submit.sh ${INPUT_PATH} ${OUTPUT_PATH} ${IMGS_PATH} ${i} 5000 ${MODE}

exit
