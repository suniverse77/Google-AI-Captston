# train for zero-shot model

CUDA_VISIBLE_DEVICES=0 python main.py \
 --clip_weight "./weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt" \
 --weight_path ./weight \
 --data_dir ./data \
 --dataset mvtec \
 --log_dir ./train_log \
 --epochs 4
