#!/bin/bash 
#SBATCH --job-name=eval_gpt_review
#SBATCH --account=passaglia
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/passaglia/slurm/R-%x.%j.out
#SBATCH --error=/home/passaglia/slurm/R-%x.%j.err

source ~/.bashrc 

conda activate jrank

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

EVAL_DIR=/home/passaglia/projects/jrank/elo

OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/eval_gpt_review.py \
 -q $EVAL_DIR/questions/rakuda_koukou_v0.jsonl \
 -p $EVAL_DIR/prompts/rakuda_prompt_threeclass.jsonl \
 -r $EVAL_DIR/prompts/rakuda_reviewer.jsonl \
 -a $EVAL_DIR/answers/rakuda_koukou_v0/gpt3.jsonl $EVAL_DIR/answers/rakuda_koukou_v0/rinna-gpt-neox-3.6b.jsonl \
 -o $EVAL_DIR/matchups/rakuda_koukou_v0 \
 -m gpt-3.5-turbo-0301 \
 -l 2 \
 -id question_id > $OUTPUT_DIR/eval_gpt_review.out 2> $OUTPUT_DIR/eval_gpt_review.err
