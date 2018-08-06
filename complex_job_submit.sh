#!/bin/bash
#PBS -S /bin/bash
#PBS -N minerva_network_topology_analysis
#PBS -j oe
#PBS -o /data/jhamer/minerva_output/
#PBS -l walltime=24:00:00
#PBS -A minervaG
#PBS -q amd32

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"

cd /home/jhamer

export INPUT_PATH=$1
export OUTPUT_PATH=$2
export START_INDEX=$3
export END_INDEX=$4
export MODE=$5

singularity exec network_topology.simg python3 MINERvA_NOvA_network_analysis/get_complex_attributes.py ${INPUT_PATH} ${OUTPUT_PATH} ${START_INDEX} ${END_INDEX} ${MODE}

exit