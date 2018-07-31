#!/bin/bash
#PBS -S /bin/bash
# -N minerva_network_topology_analysis
#-j oe
#-o /data/jhamer/minerva_output/minerva-simple-log.txt
#-l nodes=1:amd32,walltime=00:05:00
#-A minervaG
#-q amd32

echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"

cd /home/jhamer

export INPUT_PATH='/data/jhamer/minerva_networks/networks'
export OUTPUT_PATH='/data/jhamer/minerva_output'
export START_INDEX='1'
export END_INDEX='50'
export MODE='minerva'

singularity exec network_topology.simg cd MINERvA_NOvA_network_analysis/ &&\
python3 get_simple_attributes.py ${INPUT_PATH} ${OUTPUT_PATH} ${START_INDEX} ${END_INDEX} ${MODE}