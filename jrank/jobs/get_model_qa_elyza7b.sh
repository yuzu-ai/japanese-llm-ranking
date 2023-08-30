EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path elyza/ELYZA-japanese-Llama-2-7b-instruct \
    --conv_template $EVAL_DIR/templates/elyza.json \
    --model_id elyza/ELYZA-japanese-Llama-2-7b-instruct \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/elyza-7b-instruct.jsonl \
    --load_8bit False \
    --max_tokens 1024 \
    --repetition_penalty 1.05 > $OUTPUT_DIR/elyza-7b-instruct.out 2> $OUTPUT_DIR/elyza-7b-instruct.err
