#!/bin/bash

INPUT_PATH=$1
OUTPUT_PATH=$2
IMGS_PATH=$3
MODE=$4
GEN_PER_BATCH=$5

((i=1))
((end=i + GEN_PER_BATCH -1))
while [ ${end} -lt 5000 ]
do
    cd /home/jhamer/MINERVA_NOvA_network_analysis
    qsub complex_job_submit.sh ${INPUT_PATH} ${OUTPUT_PATH} ${IMGS_PATH} ${i} ${end} ${MODE}
    ((i = i + GEN_PER_BATCH))
    ((end = i + GEN_PER_BATCH - 1))
done

complex_job_submit.sh ${INPUT_PATH} ${OUTPUT_PATH} ${IMGS_PATH} ${i} 5000 ${MODE}

exit
