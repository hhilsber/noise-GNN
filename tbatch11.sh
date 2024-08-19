#!/bin/bash 
#SBATCH --job-name=tj11              # Job name
#SBATCH --mem=24g                         # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=24:00:00                   # Time limit days-hrs:min:sec
#SBATCH --error=../out_batch/slurm11.err                # Error file name
#SBATCH --output=../out_batch/slurm11.out               # Output file name

source /opt/conda/etc/profile.d/conda.sh      # initialize conda
conda activate pyl                  # load up the conda environment
python main.py -config config/config_test_s3.yml