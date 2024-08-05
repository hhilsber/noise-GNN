#!/bin/bash 
#SBATCH --job-name=test_job4              # Job name
#SBATCH --mem=32g                         # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=30:00:00                   # Time limit days-hrs:min:sec
#SBATCH --error=../out_batch/slurm4.err                # Error file name
#SBATCH --output=../out_batch/slurm4.out               # Output file name

source /opt/conda/etc/profile.d/conda.sh      # initialize conda
conda activate pyl                  # load up the conda environment
python main.py -config config/config_test.yml                      # launch the code