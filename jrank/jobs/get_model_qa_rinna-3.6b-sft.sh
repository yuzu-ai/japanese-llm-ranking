#!/bin/bash 
#SBATCH --job-name=qa_rinna
#SBATCH --account=passaglia
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/passaglia/slurm/R-%x.%j.out
#SBATCH --error=/home/passaglia/slurm/R-%x.%j.err

source ~/.bashrc
conda activate jrank

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

EVAL_DIR=/home/passaglia/projects/jrank/jrank

OUTPUTS_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path rinna/japanese-gpt-neox-3.6b-instruction-sft-v2 \
    --conv_template $EVAL_DIR/templates/rinna.json \
    --model_id rinna/japanese-gpt-neox-3.6b-instruction-sft-v2 \
    --question_file $EVAL_DIR/questions/rakuda_koukou_v0.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_koukou_v0/rinna-sft.jsonl \
    --load_8bit False \
    --max_new_tokens 256 \
    --repetition_penalty 1.1 > $OUTPUTS_DIR/rinna-sft.out 2> $OUTPUTS_DIR/rinna-sft.err
