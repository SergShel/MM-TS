#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/mm-ts/mm-ts-baseline.out
#SBATCH -e /dais/fs/scratch/dduka/logs/mm-ts/mm-ts-baseline.err

#SBATCH -J mm-ts-baseline
#SBATCH --time=23:59:59

#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=1

#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000

module purge
micromamba activate avion_fa2

nvidia-smi

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Job running on nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES" 
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

EXP_NAME="mm-ts-baseline"
EXP_PATH="/dais/fs/scratch/dduka/training_metadata/${EXP_NAME}"

mkdir -p $EXP_PATH

cd /u/dduka/project/mm-ts-avion

PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=4 scripts/main_lavila_finetune_mir.py \
    --root datasets/EK100/EK100_320p_15sec_30fps_libx264/video_320p_15sec \
    --train-metadata datasets/EK100/EK100_320p_15sec_30fps_libx264/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv \
    --val-metadata datasets/EK100/EK100_320p_15sec_30fps_libx264/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv \
    --relevancy-path datasets/EK100/EK100_320p_15sec_30fps_libx264/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl \
    --video-chunk-length 15 \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 128 \
    --fused-decode-crop \
    --use-multi-epochs-loader \
    --pretrain-model experiments/avion_pretrain_lavila_vitb_best.pt \
    --loss-type clip \
    --enable-wandb \
    --wandb-project mm-ts-avion \
    --wandb-entity dduka-max-planck-society \
    --wandb-run-name ${EXP_NAME} \
    --wandb-log-freq 10 \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt \