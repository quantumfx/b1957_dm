#!/bin/bash -l
#PBS -q workq
#PBS -r n
#PBS -l nodes=1:ppn=8,walltime=12:00:00
#PBS -N B1957DM
 
# load modules (must match modules used for compilation)
module purge
#module load gcc/7.3.0 python/3.6.4
module load intel/intel-18 python/3.6-mkl
 
# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd $PBS_O_WORKDIR
filelist="filelist-2014-06-15-E1E2.txt"
basefolder="dm_2014-06-15"
bin_factor=$PBS_ARRAYID
outfolder=$basefolder/$bin_factor\_ftfit_multimin_NM/

mkdir -p $outfolder

# EXECUTION COMMAND; python fold_dm.py (pulses to fold) (data list) (output folder)
python3 fold_dm_nm.py $bin_factor $filelist $outfolder
