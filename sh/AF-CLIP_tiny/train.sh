# train for zero-shot model

# clip_weight: CLIP 모델 가중치 파일
# weight_weight: Adaptor/Prompt 가중치 경로 (없으면 None)
# data_dir: 데이터셋 경로
# dataset: 학습 데이터셋 (mvtec or visa)
# dataset_list: 평가 데이터셋 목록
# result_dir: 결과 저장 폴더
# vis: 시각화 유무 (0이면 학습 및 평가 / 1이면 시각화)
# 

run_name='vit-61m_text-29m_manual'

CUDA_VISIBLE_DEVICES=0 python main.py \
 --clip_weight ./weight/${run_name}/TinyCLIP-ViT-61M-32-Text-29M-LAION400M.pt \
 --data_dir ./data \
 --dataset visa \
 --dataset_list mvtec visa \
 --result_dir ./results/${run_name} \
 --epochs 4 \
 --vis 0 \
# --weight_path ./weight/${run_name}
