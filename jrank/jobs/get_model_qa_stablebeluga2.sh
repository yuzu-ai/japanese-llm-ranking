EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/get_model_qa.py \
    --model_path stabilityai/StableBeluga2 \
    --conv_template $EVAL_DIR/templates/stablebeluga2.json \
    --model_id stabilityai/StableBeluga2 \
    --question_file $EVAL_DIR/questions/rakuda_v1.jsonl \
    --answer_file $EVAL_DIR/answers/rakuda_v1/stablebeluga2.jsonl \
    --load_8bit True \
    --num_gpus 2 \
    --repetition_penalty 1.0 > $OUTPUT_DIR/stablebeluga2.out 2> $OUTPUT_DIR/stablebeluga2.err
