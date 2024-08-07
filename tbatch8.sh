#!/bin/bash 
#SBATCH --job-name=tj8              # Job name
#SBATCH --mem=24g                         # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=18:00:00                   # Time limit days-hrs:min:sec
#SBATCH --error=../out_batch/slurm8.err                # Error file name
#SBATCH --output=../out_batch/slurm8.out               # Output file name

source /opt/conda/etc/profile.d/conda.sh      # initialize conda
conda activate pyl                  # load up the conda environment
python main.py -config config/config_arxiv9.yml
python main.py -config config/config_arxiv10.yml