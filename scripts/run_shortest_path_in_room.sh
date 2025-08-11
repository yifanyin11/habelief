#!/bin/bash

source /home/ubuntu/VLMP/tianmin-project/miniconda3/etc/profile.d/conda.sh
conda activate dfm-pixel-habitat
export CUDA_VISIBLE_DEVICES=7

nvidia-smi

python /home/ubuntu/VLMP/tianmin-project/yyin34/codebase/embodied_tasks/habelief/habitat_llm/examples/shortest_path_in_room.py

conda deactivate

