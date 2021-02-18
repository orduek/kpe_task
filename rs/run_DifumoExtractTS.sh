#!/bin/bash
#SBATCH --partition=general
#SBATCH --output=extractTS_ses2.txt
#SBATCH --error=extractTS_ses2.err
#SBATCH --job-name=extractTS_ses2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=or.duek@yale.edu

echo "Running script"



module load miniconda

source activate py37_dev

python /home/oad4/kpe_task/rs/difumo_extract_timeline.py
