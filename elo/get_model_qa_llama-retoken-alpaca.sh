#!/bin/bash 
#SBATCH --job-name=qa_llama_retoken
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

EVAL_DIR=/home/passaglia/projects/jrank/eval
CHECKPOINT_DIR=/home/passaglia/projects/llama-retoken/checkpoints

BASE_MODEL_DIR=$CHECKPOINT_DIR/llama7b-gptsan-embeddingtuned-wikipedia-alpaca
LORA_DIR=$CHECKPOINT_DIR/llama7b-gptsan-embeddingtuned-wikipedia-alpaca-lora-alpaca
OUTPUTS_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path $BASE_MODEL_DIR \
    --lora_path $LORA_DIR \
    --conversation_config $EVAL_DIR/gptsan-conversation_config.json \
    --model_id "llama-retoken" \
    --question_file $EVAL_DIR/questions/rakuda_koukou_v0.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_koukou_v0/llama_retoken_alpaca_gpt4.jsonl \
    --max_new_tokens 512 \
    --load_8bit False  > $OUTPUTS_DIR/llama-retoken-alpaca.out 2> $OUTPUTS_DIR/llama-retoken-alpaca.err
