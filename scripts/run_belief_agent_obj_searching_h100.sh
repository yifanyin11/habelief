#!/bin/bash

source /home/ubuntu/VLMP/tianmin-project/miniconda3/etc/profile.d/conda.sh
conda activate dfm-pixel-habitat

nvidia-smi

# python /home/ubuntu/VLMP/tianmin-project/yyin34/codebase/embodied_tasks/habelief/habitat_llm/examples/object_searching/seq/belief_agent_object_searching.py

python /home/ubuntu/VLMP/tianmin-project/yyin34/codebase/embodied_tasks/habelief/habitat_llm/examples/object_searching/dfm/belief_agent_object_searching_vlm.py

conda deactivate

