EVAL_DIR=/home/mdxuser/japanese-llm-ranking/jrank
OUTPUT_DIR=$EVAL_DIR

python $EVAL_DIR/get_super-torin_qa.py -q $EVAL_DIR/questions/rakuda_v1.jsonl -o $EVAL_DIR/answers/rakuda_v1/super-torin.jsonl > $OUTPUT_DIR/qa_torin.out 2> $OUTPUT_DIR/qa_torin.err
