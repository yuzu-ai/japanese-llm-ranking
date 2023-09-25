EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUTS_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path rinna/japanese-gpt-neox-3.6b-instruction-ppo \
    --conv_template $EVAL_DIR/templates/rinna.json \
    --model_id rinna/japanese-gpt-neox-3.6b-instruction-ppo \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/rinna-ppo.jsonl \
    --load_8bit False \
    --max_new_tokens 256 \
    --repetition_penalty 1.1 > $OUTPUTS_DIR/rinna-ppo.out 2> $OUTPUTS_DIR/rinna-ppo.err
