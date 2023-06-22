#!/bin/bash 
#SBATCH --job-name=qa_gpt4alpaca
#SBATCH --account=passaglia
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/passaglia/projects/llama-retoken/slurm/R-%x.%j.out
#SBATCH --error=/home/passaglia/projects/llama-retoken/slurm/R-%x.%j.err

source ~/.bashrc
conda activate jrank

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

EVAL_DIR=/home/passaglia/projects/jrank/jrank
CHECKPOINT_DIR=/home/passaglia/projects/llama-retoken/checkpoints

BASE_MODEL_DIR=$CHECKPOINT_DIR/llama7b
LORA_DIR=$CHECKPOINT_DIR/llama7b-gpt4alpaca
OUTPUTS_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path $BASE_MODEL_DIR \
    --lora_path $LORA_DIR \
    --model_id "alpaca" \
    --question_file $EVAL_DIR/questions/rakuda_koukou_v0.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_koukou_v0/gpt4alpaca.jsonl \
    --max_new_tokens 512 \
    --load_8bit False  > $OUTPUTS_DIR/gpt4alpaca.out 2> $OUTPUTS_DIR/gpt4alpaca.err
