#!/bin/bash
#SBATCH --job-name=jpt_a6000
#SBATCH --time=1000-00:00:00
#SBATCH --partition=A6000
#SBATCH --gres=gpu:2
#SBATCH --nodelist=chili
#SBATCH --cpus-per-task=10
#SBATCH --mem=50GB
#SBATCH --output=/data/vision/polina/scratch/wpq/github/cxrpe/scripts/jpt_a6000.out

# Customize this line to point to conda installation
path_to_conda="/data/vision/polina/shared_software/miniconda3"
# huggingface cache
export HF_HOME='/data/vision/polina/scratch/wpq/huggingface_cache'

source ${path_to_conda}/bin/activate cxrpe
cd /data/vision/polina/scratch/wpq/github/
jupyter notebook --no-browser --port=8891


