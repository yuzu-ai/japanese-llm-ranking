EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path line-corporation/japanese-large-lm-3.6b-instruction-sft \
    --model_id line-corporation/japanese-large-lm-3.6b-instruction-sft \
    --conv_template $EVAL_DIR/templates/line.json \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/line-3.6b-sft.jsonl \
    --num_gpus 1 \
    --max_tokens 512 \
    --temperature 0.7 \
    --repetition_penalty 1.1 \
    --top_p 0.90 > $OUTPUT_DIR/line-3.6b-sft.out 2> $OUTPUT_DIR/line-3.6b-sft.err
