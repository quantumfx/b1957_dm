#!/bin/bash -l
#PBS -q workq
#PBS -r n
#PBS -l nodes=1:ppn=8,walltime=12:00:00
#PBS -N B1957DM
 
# load modules (must match modules used for compilation)
module purge
module load gcc/7.3.0 python/3.6.4
 
# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd $PBS_O_WORKDIR
 
# EXECUTION COMMAND; python wholedaydm.py (pulses to fold) (data) (output folder)
python3 fold_dm.py 40 filelist-2014-06-15-E1E2.txt dm_2014-06-15/40/
