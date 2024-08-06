#!/bin/bash 
#SBATCH --job-name=test_job5              # Job name
#SBATCH --mem=24g                         # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=36:00:00                   # Time limit days-hrs:min:sec
#SBATCH --error=../out_batch/slurm5.err                # Error file name
#SBATCH --output=../out_batch/slurm5.out               # Output file name

source /opt/conda/etc/profile.d/conda.sh      # initialize conda
conda activate pyl                  # load up the conda environment
python main.py -config config/config_arxiv5.yml
python main.py -config config/config_arxiv6.yml
python main.py -config config/config_arxiv7.yml
python main.py -config config/config_arxiv8.yml