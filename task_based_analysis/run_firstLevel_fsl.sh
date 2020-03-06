#!/bin/bash
#SBATCH --partition=general
#SBATCH --output=firstLevelFSLlog.txt
#SBATCH --error=firstLevelFSLerror.err
#SBATCH --job-name=firstLevelAnalysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --time=16:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=or.duek@yale.edu

echo "Running script"



module load miniconda

source activate py37_dev

python /home/oad4/kpe_task/task_based_analysis/fmri_fsl_cluster.py
