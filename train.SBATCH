#!/bin/bash
#SBATCH --job-name=animegan_torch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12GB
#SBATCH --time=120:00:00
#SBATCH --mail-type=END
#SBATCH --output=slurm_a10.out
#SBATCH --error=slurm_a10.err
#SBATCH --gres=gpu:a10:1 # How much gpu need, n is the number
#SBATCH -p aquila

module purge
module load miniconda
module load cuda

echo "start training"

cd /gpfsnyu/scratch/jj2915/Capstone-AnimeGAN-Framework
source activate torch_env
python train.py --dataset Hayao --batch-size 8
conda deactivate

echo "end training"
