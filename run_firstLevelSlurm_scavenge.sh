#!/bin/bash
#SBATCH --partition scavenge
#SBATCH --job-name=firstLevelAnalysisScavenge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40g
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=or.duek@yale.edu

echo "Running script"

#echo "Loading FSL"
#module load FSL 

#echo "Loading miniconda"
#module load miniconda 
#echo "activating env"

#source activate py37_dev 
#cd /home/oad4/kpe_task

#./firstSecondLevel_slurm.py

module load FSL
module load miniconda
source activate py37_dev
cd /home/oad4/kpe_task/
./firstSecondLevel_slurm.py 

