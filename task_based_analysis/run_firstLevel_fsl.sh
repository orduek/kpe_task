#!/bin/bash
#SBATCH --partition=general
#SBATCH --output=firstLevelKPE_ses3.txt
#SBATCH --error=firstLevel_KPE_ses3.err
#SBATCH --job-name=firstLevelKPE3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10G
#SBATCH --time=35:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=or.duek@yale.edu

echo "Running script"



module load miniconda

source activate py37_dev

python /home/oad4/kpe_task/task_based_analysis/fmri_fsl_cluster.py
