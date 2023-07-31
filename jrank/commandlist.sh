export CUDA_VISIBLE_DEVICES=0,1
bash ./jobs/get_model_qa_stablebeluga2.sh
export CUDA_VISIBLE_DEVICES=0
bash ./jobs/get_model_qa_rwkv-v1.sh
export CUDA_VISIBLE_DEVICES=0,1
bash ./jobs/get_model_qa_mpt-30b-chat.sh


# conda activate jrank 
# nohup bash commandlist.sh &