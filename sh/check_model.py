import os
from collections import defaultdict

import torch

weight_name = "vit-8m_text-3m_manual"
file_name = "TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt"
weight_path = os.path.join("AF-CLIP_tiny/weight", weight_name, file_name)

checkpoint = torch.load(weight_path, map_location="cpu")
model_weights = checkpoint['state_dict']

# 모델의 모든 layer의 파라미터 출력
def all_layer_params(model_weights):
    total_params = 0  # 전체 파라미터 개수를 합산할 변수

    # model_weights 딕셔너리의 키(key)와 값(value)을 순회합니다.
    for key, value in model_weights.items():
        
        # value는 torch.Tensor 입니다.
        # .numel()은 텐서의 총 원소(파라미터) 개수를 반환합니다.
        num_params = value.numel()
        
        # 총 파라미터 개수에 더합니다.
        total_params += num_params
        
        # 텐서의 형태(shape)와 파라미터 개수를 함께 출력합니다.
        print(f"[{key}]")
        print(f"  Shape: {list(value.shape)}")
        print(f"  Parameters: {num_params:,}") # {num_params:,}는 천 단위 쉼표를 추가해 줍니다.
        print("-" * 20)

    print("=================================")
    print(f"총 파라미터 개수 (Total Parameters): {total_params:,}")

# 블록별로 파라미터 출력
def block_params(model_weights, BLOCK_DEPTH=6):
    # 2. 파라미터를 블록별로 집계할 딕셔너리
    # defaultdict(int)는 키가 없을 때 자동으로 0을 기본값으로 생성해 줍니다.
    block_params = defaultdict(int)
    total_params = 0

    # 3. 모든 가중치를 순회하며 블록별로 합산
    for key, value in model_weights.items():
        num_params = value.numel()
        total_params += num_params
        
        parts = key.split('.')
        
        # 4. 블록 이름 결정
        block_name = ""
        if len(parts) > BLOCK_DEPTH:
            # 키가 설정한 깊이보다 길면, 깊이만큼 잘라서 블록 이름으로 사용
            # 예: ...resblocks.0.mlp.c_fc.weight -> ...resblocks.0.mlp
            block_name = '.'.join(parts[:BLOCK_DEPTH])
        else:
            # 키가 설정한 깊이보다 짧거나 같으면 (예: ...final_ln.weight)
            # .weight, .bias를 제외한 부모 모듈을 블록 이름으로 사용
            block_name = '.'.join(parts[:-1]) 
            
        # 5. 해당 블록에 파라미터 개수 누적
        block_params[block_name] += num_params

    # 6. 결과 출력
    print(f"--- 블록별 파라미터 개수 (Depth={BLOCK_DEPTH}) ---")

    # (선택) 가독성을 위해 블록 이름 순으로 정렬
    sorted_block_names = sorted(block_params.keys())

    for block_name in sorted_block_names:
        params = block_params[block_name]
        print(f"[{block_name}]")
        print(f"  Parameters: {params:,}")
        print("-" * 20)

    print("=================================")
    print(f"총 파라미터 개수 (Total Parameters): {total_params:,}")

if __name__ == '__main__':
    #all_layer_params(model_weights)
    block_params(model_weights)