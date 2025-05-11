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

export MASTER_PORT=$((12000 + RANDOM % 1000))

torchrun  --nnodes 1 --nproc_per_node 1 --master_port $MASTER_PORT\
    /scratch/tshu2/yyin34/projects/3d_belief/embodied_belief/DFM/experiment_scripts/train_3D_diffusion_pixel_epi_temporal.py \
    dataset=clevr_seq \
    setting_name=pixelsplat_h100 \
    name=pixelsplat \
    stage=sameshape \
    results_folder="/scratch/tshu2/yyin34/projects/3d_belief/DFM/outputs/training/pixelsplat_temporal/clevr/test" \
    ngpus=1 \
    image_size=64 \
    ctxt_min=15 \
    ctxt_max=16 \
    model/encoder=epidiff \
    model.encoder.use_epipolar_transformer=false \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=true \
    model.encoder.evolve_ctxt=false \
    model/encoder/backbone=dit \
    model.encoder.backbone.use_diff_pos_embed=true \
    model_type=dit \
    clean_target=false \
    use_identity=true \
    intermediate=true \
    load_optimizer=false \
    num_intermediate=10 \
    adjacent_angle=0.589

python /scratch/tshu2/yyin34/projects/3d_belief/embodied_belief/DFM/experiment_scripts/temporal_inference_pixel_epi.py \
    dataset=clevr_seq \
    batch_size=1 \
    num_target=1 \
    num_context=1 \
    stage=colors \
    model/encoder=epidiff \
    model.encoder.use_epipolar_transformer=false \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=false \
    model.encoder.evolve_ctxt=false \
    model/encoder/backbone=dit \
    model.encoder.backbone.use_diff_pos_embed=true \
    model_type=dit \
    temperature=0.85 \
    sampling_steps=50 \
    name=pixel_inference \
    image_size=64 \
    inferece_sample_from_dataset=true \
    inference_num_samples=20 \
    inference_min_frames=15 \
    inference_max_frames=16 \
    clean_target=false \
    clevr_first_frame_prob=1.0 \
    clevr_start_frame_id=1 \
    adjacent_angle=0.589 \
    checkpoint_path=/scratch/tshu2/yyin34/projects/3d_belief/DFM/outputs/weights/pixelsplat_temporal/clevr/colors/model-26.pt \
    results_folder=/scratch/tshu2/yyin34/projects/3d_belief/DFM/outputs/inference/pixelsplat_temporal/clevr/colors

conda deactivate

