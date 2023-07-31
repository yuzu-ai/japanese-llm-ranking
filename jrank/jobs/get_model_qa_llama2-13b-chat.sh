EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path meta-llama/Llama-2-13b-chat-hf \
    --conv_template $EVAL_DIR/templates/llama2-chat-ja.json \
    --model_id meta-llama/Llama-2-13b-chat-hf \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/llama2-13b-chat.jsonl \
    --load_8bit False \
    --max_new_tokens 1024 \
    --repetition_penalty 1.0 > $OUTPUT_DIR/llama2-13b-chat.out 2> $OUTPUT_DIR/llama2-13b-chat.err
