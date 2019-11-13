#!/bin/bash
#SBATCH --partition=scavenge
#SBATCH --output=firstLevel_log.txt
#SBATCH --error=firstLevel_error.err
#SBATCH --job-name=firstLevelAnalysisScavenge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=30g
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=or.duek@yale.edu

echo "Running script"

module load miniconda
module load FSL

source activate py37_dev
export OPENBLAS_NUM_THREADS=1
python /home/oad4/kpe_task/firstSecondLevel_slurm.py
