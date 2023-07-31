EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path mosaicml/mpt-30b-chat \
    --conv_template mpt-30b-chat \
    --model_id mosaicml/mpt-30b-chat \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/mpt-30b-chat.jsonl \
    --load_8bit False \
    --num_gpus 2 \
    --max_tokens 2048 \
    --repetition_penalty 1.0 > $OUTPUT_DIR/mpt-30b-chat.out 2> $OUTPUT_DIR/mpt-30b-chat.err
