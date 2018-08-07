#!/bin/bash

INPUT_PATH=$1
OUTPUT_PATH=$2
IMGS_PATH=$3
MODE=$4
GEN_PER_BATCH=$5

((i = 1))
((start = 1000))
((end = i + GEN_PER_BATCH -1))
while [ ${i} -le 23 ]
do
    if [ ${i} < 10 ]
    then
    IMGS_PATH=${IMGS_PATH}/0${i}/hadmultkineimgs_127x94_me1Amc.hdf5
    else
    IMGS_PATH=${IMGS_PATH}/${i}/hadmultkineimgs_127x94_me1Amc.hdf5
    fi

    cd /home/jhamer/MINERvA_NOvA_network_analysis
    qsub complex_job_submit.sh ${INPUT_PATH} ${OUTPUT_PATH} ${IMGS_PATH} ${start} ${end} ${MODE}
    ((i = i + 1))
    ((start = start + GEN_PER_BATCH))
    ((end = start + GEN_PER_BATCH - 1))
done

exit
