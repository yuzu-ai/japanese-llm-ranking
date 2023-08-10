EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path stabilityai/japanese-stablelm-instruct-alpha-7b \
    --model_id stabilityai/japanese-stablelm-instruct-alpha-7b \
    --conv_template $EVAL_DIR/templates/japanese-stablelm.json \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/japanese-stablelm-instruct-alpha.jsonl \
    --num_gpus 1 \
    --max_tokens 512 \
    --temperature 1 \
    --repetition_penalty 1.0 \
    --top_p 0.95 > $OUTPUT_DIR/japanese-stablelm.out 2> $OUTPUT_DIR/japanese-stablelm.err
