# train for zero-shot model
data_dir=./data

CUDA_VISIBLE_DEVICES=0 python main.py \
 --model "ViT-L/14@336px" \
 --log_dir ./train_log \
 --dataset mvtec \
 --data_dir $data_dir
