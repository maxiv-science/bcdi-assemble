#!/bin/bash
#
# job time, change for what your job requires
#SBATCH -t 10:00:00
#
# job name
#SBATCH -J pynx
#SBATCH -N 1
#SBATCH -p v100

# write this script to stdout-file - useful for scripting errors
echo $PWD
echo $(date)

echo "*******************************************************"
cat $0
echo "*******************************************************"

source /mxn/home/alebjo/analysis/assembly/source_me

rm -rf raw
mkdir raw

rm *.cxi
pynx-id01cdi.py data=prepared_10.npz nb_run_keep=20 nb_run=1000 support_threshold=0.1,0.5
pynx-cdi-analysis.py *.cxi modes
mv *.cxi raw/
mv modes.h5 modes_10.h5

rm *.cxi
pynx-id01cdi.py data=prepared_1.npz nb_run_keep=20 nb_run=1000 support_threshold=0.1,0.5
pynx-cdi-analysis.py *.cxi modes
mv *.cxi raw/
mv modes.h5 modes_1.h5

rm *.cxi
pynx-id01cdi.py data=prepared_0.npz nb_run_keep=20 nb_run=1000 support_threshold=0.1,0.5
pynx-cdi-analysis.py *.cxi modes
mv *.cxi raw/
mv modes.h5 modes_0.h5
