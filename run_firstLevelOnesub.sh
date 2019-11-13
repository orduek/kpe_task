#!/bin/bash
#SBATCH --partition=general
#SBATCH --output=firstLevel_1253log.txt
#SBATCH --error=firstLevel_1253error.err
#SBATCH --job-name=firstLevelAnalysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=15G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=or.duek@yale.edu

echo "Running script"


module load FSL/5.0.9-centos6_64
module load miniconda

source activate py37_dev

python /home/oad4/kpe_task/firstSecondLevel.py
