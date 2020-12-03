#!/bin/bash
#SBATCH --partition=general
#SBATCH --output=firstLevelSPM_ses3.txt
#SBATCH --error=firstLevelSPM_ses3.err
#SBATCH --job-name=SPMfirstLevelAnalysis_ses3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --time=13:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=or.duek@yale.edu

echo "Running script"



module load miniconda
module load MATLAB/2019b
source activate py37_dev

python /home/oad4/kpe_task/task_based_analysis/runSPM_cli.py
