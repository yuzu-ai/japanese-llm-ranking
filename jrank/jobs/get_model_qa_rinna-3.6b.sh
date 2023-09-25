EVAL_DIR=/home/passaglia/projects/jrank/jrank
OUTPUTS_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path rinna/japanese-gpt-neox-3.6b \
    --conv_template $EVAL_DIR/templates/stormy.json \
    --model_id rinna/japanese-gpt-neox-3.6b \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/rinna.jsonl \
    --load_8bit False \
    --max_new_tokens 256 \
    --repetition_penalty 1.1 > $OUTPUTS_DIR/rinna.out 2> $OUTPUTS_DIR/rinna.err
