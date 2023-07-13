python get_model_qa.py \
    --model_path models/RWKV-4-World-JPNtuned-7B-v1-OnlyForTest_55%_trained-20230711-ctx4096 \
    --model_id rwkv-world-jpn-55 \
    --conv_template templates/rwkv_world.json \
    --question_file questions/rakuda_v1.jsonl \
    --answer_file answers/rakuda_v1/rwkv.jsonl \
    --max_new_tokens 512 \
    --temperature 1.0 \
    --top_p 0.2 \
    --load_8bit False  > rwkv.out 2> rwkv.err
