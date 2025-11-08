# train for zero-shot model
data_dir=./data

CUDA_VISIBLE_DEVICES=0 python main.py \
 --model_weight "./weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt"\
 --model "ViT-L/14@336px" \
 --log_dir ./train_log \
 --dataset mvtec \
 --data_dir $data_dir
