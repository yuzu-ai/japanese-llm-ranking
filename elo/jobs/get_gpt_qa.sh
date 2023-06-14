#!/bin/bash 
#SBATCH --job-name=eval_gpt
#SBATCH --account=passaglia
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/passaglia/slurm/R-%x.%j.out
#SBATCH --error=/home/passaglia/slurm/R-%x.%j.err

source ~/.bashrc 

conda activate jrank

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

EVAL_DIR=/home/passaglia/projects/jrank/elo

OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/get_gpt_qa.py -q $EVAL_DIR/questions/rakuda_koukou_v0.jsonl -o $EVAL_DIR/answers/rakuda_koukou_v0/gpt3.jsonl > $OUTPUT_DIR/qa_gpt.out 2> $OUTPUT_DIR/qa_gpt.err
