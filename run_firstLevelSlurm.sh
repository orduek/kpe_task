#!/bin/bash
#SBATCH --partition=general
#SBATCH --output=firstLevel_log.txt
#SBATCH --error=firstLevel_error.err
#SBATCH --job-name=firstLevelAnalysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=or.duek@yale.edu

echo "Running script"

module load miniconda
module load FSL

source activate py37_dev
python /home/oad4/kpe_task/firstSecondLevel_slurm.py
