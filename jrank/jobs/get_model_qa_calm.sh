EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUTS_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path cyberagent/open-calm-7b \
    --model_id cyberagent/open-calm-7b \
    --conv_template $EVAL_DIR/templates/stormy.json \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/calm.jsonl \
    --max_new_tokens 256 \
    --repetition_penalty 1.1 \
    --load_8bit False  > $OUTPUTS_DIR/calm.out 2> $OUTPUTS_DIR/calm.err
