#!/bin/bash 
#SBATCH --job-name=tj1              # Job name
#SBATCH --mem=24g                         # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=48:00:00                   # Time limit days-hrs:min:sec
#SBATCH --error=../out_batch/slurm1.err                # Error file name
#SBATCH --output=../out_batch/slurm1.out               # Output file name

source /opt/conda/etc/profile.d/conda.sh      # initialize conda
conda activate pyl                  # load up the conda environment
python main.py -config config/config_test_s.yml
python main.py -config config/config_test_s2.yml
#python main.py -config config/config_test_s3.yml