#!/bin/bash
#PBS -S /bin/bash
#PBS -N minerva_network_topology_analysis
#PBS -j oe
#PBS -o /data/jhamer/minerva_output/minerva-simple-log.out
#PBS -l walltime=04:00:00
#PBS -A minervaG
#PBS -q amd32

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"

cd /home/jhamer

export INPUT_PATH='/data/jhamer/minerva_networks/networks'
export OUTPUT_PATH='/data/jhamer/minerva_output'
export START_INDEX='1'
export END_INDEX='5000'
export MODE='minerva'

singularity exec network_topology.simg python3 MINERvA_NOvA_network_analysis/get_simple_attributes.py ${INPUT_PATH} ${OUTPUT_PATH} ${START_INDEX} ${END_INDEX} ${MODE}

exit