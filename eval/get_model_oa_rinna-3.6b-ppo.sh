#!/bin/bash 
#SBATCH --job-name=oa_rinna
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

EVAL_DIR=/home/passaglia/projects/jrank/eval

OUTPUTS_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_oa.py \
    --model_path rinna/japanese-gpt-neox-3.6b-instruction-ppo \
    --conversation_config $EVAL_DIR/rinna-conversation_config.json \
    --model_id "rinna" \
    --question_file $EVAL_DIR/questions/oa_ja.jsonl \
    --answer_file $EVAL_DIR/answers/oa_rinna-gpt-neox-3.6b.jsonl \
    --load_8bit False \
    --repetition_penalty 1.1 > $OUTPUTS_DIR/rinna-gpt-neox-3.6b.out 2> $OUTPUTS_DIR/rinna-gpt-neox-3.6b.err
