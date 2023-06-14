#!/bin/bash 
#SBATCH --job-name=qa_story
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
export CUDA_VISIBLE_DEVICES=2

EVAL_DIR=/home/passaglia/projects/jrank/elo
CHECKPOINT_DIR=/home/passaglia/projects/llama-retoken/checkpoints

LORA_DIR=$CHECKPOINT_DIR/opencalm7b-alpaca-colab-4400
OUTPUTS_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path cyberagent/open-calm-7b \
    --lora_path $LORA_DIR \
    --conversation_config $EVAL_DIR/configs/alpaca_short_jp-conversation_config.json \
    --model_id calm-alpaca \
    --question_file $EVAL_DIR/questions/rakuda_koukou_v0.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_koukou_v0/calm-alpaca.jsonl \
    --max_new_tokens 256 \
    --repetition_penalty 1.1 \
    --load_8bit False  > $OUTPUTS_DIR/calm_alpaca.out 2> $OUTPUTS_DIR/calm_alpaca.err
