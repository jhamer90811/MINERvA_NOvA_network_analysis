#!/bin/bash
#PBS -S /bin/bash
#PBS -N minerva_network_topology_analysis
#PBS -j oe
#PBS -o /data/jhamer/minerva_output/
#PBS -l walltime=24:00:00
#PBS -A minervaG
#PBS -q intel12

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"

cd /home/jhamer

INPUT_PATH=$1
OUTPUT_PATH=$2
IMG_PATH=$3
START_INDEX=$4
END_INDEX=$5
MODE=$6

singularity exec network_topology.simg python3 MINERvA_NOvA_network_analysis/get_complex_attributes.py ${INPUT_PATH} ${OUTPUT_PATH} ${IMG_PATH} ${START_INDEX} ${END_INDEX} ${MODE}

exit