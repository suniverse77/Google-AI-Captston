
# train for zero-shot model
data_dir=./data
CUDA_VISIBLE_DEVICES=0 python main.py --log_dir ./train_log --dataset mvtec --data_dir ./data --weight ./weight
