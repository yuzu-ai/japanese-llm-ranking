EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path matsuo-lab/weblab-10b-instruction-sft \
    --model_id matsuo-lab/weblab-10b-instruction-sft \
    --conv_template $EVAL_DIR/templates/weblab.json \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/weblab-10b-instruction-sft.jsonl \
    --num_gpus 1 \
    --max_tokens 512 \
    --temperature 0.5 \
    --repetition_penalty 1.05 \
    --top_p 0.90 > $OUTPUT_DIR/weblab-10b-instruction-sft.out 2> $OUTPUT_DIR/weblab-10b-instruction-sft.err
