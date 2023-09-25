EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path $EVAL_DIR/models/RWKV-4-World-JPNtuned-7B-v1-20230718-ctx4096.pth \
    --model_id rwkv-world-jp-v1 \
    --conv_template $EVAL_DIR/templates/rwkv_world.json \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/rwkv-jp-v1-test.jsonl \
    --max_tokens 1024 \
    --temperature 1.0 \
    --top_p 0.2 \
    --load_8bit False  > $OUTPUT_DIR/rwkv-jp-v1.out 2> $OUTPUT_DIR/rwkv-jp-v1.err
