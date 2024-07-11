#!/bin/bash 
#SBATCH --job-name=test_job6              # Job name
#SBATCH --mem=32g                         # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=20:00:00                   # Time limit days-hrs:min:sec
#SBATCH --error=../out_batch/slurm6.err                # Error file name
#SBATCH --output=../out_batch/slurm6.out               # Output file name

source /opt/conda/etc/profile.d/conda.sh      # initialize conda
conda activate pyl                  # load up the conda environment
python main.py -config config/config_expe.yml                      # launch the code