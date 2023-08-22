EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path leemeng/llm-experimental \
    --model_id japanese-stablelm-experimental \
    --conv_template $EVAL_DIR/templates/japanese-stablelm.json \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/japanese-stablelm-experimental.jsonl \
    --num_gpus 1 \
    --max_tokens 512 \
    --temperature 1 \
    --repetition_penalty 1.1 \
    --top_p 0.95 > $OUTPUT_DIR/japanese-stablelm-experimental.out 2> $OUTPUT_DIR/japanese-stablelm-experimental.err
