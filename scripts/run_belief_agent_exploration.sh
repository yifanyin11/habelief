#!/bin/bash
#SBATCH --job-name=3b_train_pixelsplat
#SBATCH --output=/scratch/tshu2/yyin34/logs/3b_train_pixelsplat%j.out
#SBATCH --error=/scratch/tshu2/yyin34/logs/3b_train_pixelsplat%j.err
#SBATCH --partition=nvl
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --nodelist=n11
source /weka/scratch/tshu2/yyin34/projects/3d_belief/miniconda3/etc/profile.d/conda.sh
conda activate dfm-pixel-habitat

# /scratch/tshu2/yyin34/projects/3d_belief/scripts

nvidia-smi

python /scratch/tshu2/yyin34/projects/3d_belief/partnr-planner/habitat_llm/examples/belief_agent_exploration.py


conda deactivate

