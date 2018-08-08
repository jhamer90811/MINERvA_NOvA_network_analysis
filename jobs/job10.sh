#!/bin/bash
#PBS -S /bin/bash
#PBS -N network_topology10
#PBS -j oe
#PBS -o /data/jhamer/minerva_output/
#PBS -l walltime=24:00:00
#PBS -A minervaG
#PBS -q intel12

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"

cd /home/jhamer

INPUT_PATH=/data/jhamer/minerva_networks/networks
OUTPUT_PATH=/data/jhamer/minerva_output/complex_attributes
IMG_PATH=/lfstev/e-938/jhamer/10/hadmultkineimgs_127x94_me1Amc.hdf5
START_INDEX=3405
END_INDEX=3450
MODE='minerva'

singularity exec -B /lfstev:/lfstev network_topology.simg python3 MINERvA_NOvA_network_analysis/get_complex_attributes.py ${INPUT_PATH} ${OUTPUT_PATH} ${IMG_PATH} ${START_INDEX} ${END_INDEX} ${MODE}

exit