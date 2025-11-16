# PatchCore 원본 레포 조사 노트

## 디렉터리 구조 핵심
- `src/patchcore/patchcore.py`: `PatchCore` 모델 본체. 메모리 뱅크 구축(`fit`), 추론(`predict`), 저장/로드 로직 포함.
- `src/patchcore/common.py`: 특성 집계(`NetworkFeatureAggregator`), 전처리(`Preprocessing`), 임베딩 축소(`Aggregator`), 최근접 이웃 스코어러 등 공통 구성요소.
- `src/patchcore/backbones.py`: `timm`/`torchvision`/`pretrainedmodels` 백본 팩토리. 문자열 키 기반 모델 생성.
- `src/patchcore/sampler.py`: 메모리 축소용 샘플러(`IdentitySampler`, `RandomSampler`, `GreedyCoresetSampler` 등).
- `src/patchcore/datasets/mvtec.py`: MVTEC 데이터 파이프라인 정의. 데이터셋 항목은 `{"image", "mask", "is_anomaly"}` 구조.
- `bin/run_patchcore.py`, `bin/load_and_evaluate_patchcore.py`: 학습/추론 CLI 예제 스크립트.

## 주요 의존성
- `faiss-cpu` (또는 GPU 환경에서 `faiss-gpu`): 최근접 이웃 검색.
- `pretrainedmodels`, `timm`, `torch`, `torchvision`: 백본 로더.
- `click`: CLI 스크립트 인자 파싱.
- `scikit-image`, `scikit-learn`, `scipy`, `numpy`, `matplotlib`, `pillow`, `tqdm`: 전처리·평가·유틸리티.

### 우리 프로젝트 의존성과 비교
- 이미 포함: `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `scikit-image`, `matplotlib`, `pillow`, `tqdm`, `timm`.
- 신규 필요 가능성: `faiss-cpu` (또는 GPU 대체), `pretrainedmodels`, `click`. 추가로 PatchCore 기본 스크립트는 Python 3.7+ 기준.

## 적용 시 고려사항
- 메모리 뱅크 저장 형식: `patchcore_params.pkl` + FAISS 인덱스 (`*.faiss`).
- 입력 파이프라인: 학습단계(`fit`)는 학습 데이터 로더 전체 반복, 추론(`predict`)은 점수/마스크/GT 반환.
- 디바이스 의존: 모든 모듈은 `device` 인자를 통해 GPU/CPU 설정, 최근접 이웃은 `FaissNN` 사용 시 별도 디바이스 관리 필요.
- 스크립트 구성: CLI는 Hydra 미사용, argparse 유사 구조. 우리 프로젝트의 설정/실행 구조에 맞춰 모듈화 필요.

## 프로젝트 래퍼 설계 개요
- 클래스 위치: `src/methods/patchcore.py`, `@register_method("patchcore")` 사용.
- 기본 구조:
  - `from_config`에서 핵심 하이퍼파라미터 수집 (`backbone`, `layers`, `input_height/width`, `target_dim`, `sampler` 등).
  - `setup`에서 `patchcore.backbones.load`, `PatchCore.load` 호출. `FaissNN`, 샘플러 생성 포함.
  - `prepare_category(category, train_loader)`에서 `PatchCore.fit` 실행하여 카테고리별 메모리 뱅크 구축. 내부적으로 `self._category_state[category]`에 저장/재사용 여부 관리.
  - `forward(batch)`는 `batch["image"]`를 입력으로 `PatchCore.predict` 호출 → 리스트 반환을 `torch.Tensor`로 변환 후 `ForwardResult` 구성. 필요 시 마스크 크기 조정/정규화 옵션 제공.
- 구성 요소 모듈화:
  - 샘플러 선택: `identity`, `greedy_coreset`, `approx_greedy_coreset` 등을 config 옵션으로 매핑.
  - 최근접 이웃 설정: `faiss_on_gpu`, `faiss_num_workers`.
  - 출력 후처리: 이미지 점수는 Tensor `(B,)`, 픽셀 점수는 `(B, 1, H, W)` 형태 유지. 추가 정보(`raw_scores`, `patch_shapes`)는 `ForwardResult.extra`에 저장.
  - 사전 학습 메모리 로딩:
    - `model.pretrained.root`에 PatchCore 원본 레포의 실험 디렉터리를 지정하면 카테고리별 저장 모델(`models/mvtec_<cat>/...`)을 자동 로드.
    - `model.pretrained.prepend`가 없으면 `*_patchcore_params.pkl` 파일명을 스캔해 접두사를 추론.
    - `model.pretrained.strict: true`로 설정하면 경로가 없을 때 예외 발생, 기본은 경고 후 새로 학습.
- 입력 크기/전처리:
  - `input_shape`는 `(3, height, width)`로 config 지정, 데이터로더 전처리(`resize`, `center_crop`, `normalize`)는 Step 3에서 통합.
  - `PatchMaker` 패치 크기(`patchsize`, `patchstride`) 등은 YAML에서 조정 가능하도록 명시.
- 상태 관리:
  - `setup`은 한 번만 백본과 전처리 모듈 초기화.
  - 카테고리 전환 시 `self._current_category` 추적, 필요하면 기존 메모리/스코어러 리셋 후 `fit` 재실행.
- 예외/로깅:
  - 필수 파라미터 누락 시 `ValueError`.
  - `faiss` 미설치 시 사용자 친화적 에러 메시지 제공.

## 설정 & 데이터 파이프라인 통합 전략
- 전용 설정 파일: `configs/mvtec_patchcore.yaml`
  - `model` 섹션에 PatchCore 전용 항목 추가
    - `name: patchcore`
    - `backbone: resnet18` (예시) 및 `layers: ["layer2", "layer3"]`
    - `input_size: [3, 224, 224]`
    - `pretrain_embed_dim`, `target_embed_dim`
    - `patch: {size: 3, stride: 1}`, `nn: {k: 5, faiss_on_gpu: false, num_workers: 4}`
    - `sampler: {name: greedy_coreset, ratio: 0.1}`
    - `score_map: {resize_to: 224, normalize: true}`
  - `data.transforms` 활성화: `resize`, `center_crop`, `normalize` 값을 PatchCore 입력 기준으로 설정.
  - `data.loader.use_full_train: true`로 설정해 k-shot=0이어도 train split 전체를 메모리 구축에 사용.
  - 사전 학습 실험과 동일한 구성(예: WR50, 320 해상도)을 사용할 경우 해당 백본/입력 크기/차원/샘플러를 동일하게 맞춘 별도 프로필을 정의.
  - `evaluation` 섹션은 기존 WinCLIP과 동일하게 재사용.
- 데이터 파이프라인 변경:
  - `build_mvtec_dataloaders`에서 `config["transforms"]` 값을 읽어 `torchvision.transforms` 시퀀스로 변환하여 `MvtecDataset`에 전달.
  - 마스크 변환은 이진 마스크 유지하며, 필요 시 `resize`만 적용.
  - `train_loader`는 PatchCore `fit` 단계에서 shuffle 없이 전체 순회하도록 유지.
- 범용 설정 지원:
  - `configs/base.yaml`에 정의된 `data.transforms`를 기본값으로 삼고, PatchCore 전용 설정에서 덮어쓰도록 구현.
  - 추후 다른 메서드도 동일 메커니즘을 사용할 수 있도록 공통 유틸(`utils/config.py`)에 변환 생성 헬퍼 추가 고려.

## 실행/평가 파이프라인 확장 설계
- 모듈 임포트
  - `src/main.py` 상단에서 `import methods.patchcore` 추가하여 레지스트리 등록.
  - `ForwardResult`를 활용해 공통 경로 유지, 별도 분기 최소화.
- 모델 준비 단계
  - `build_method` 호출 후 `hasattr(model, "prepare_category")` 체크하여 존재 시 호출.
  - PatchCore용 `prepare_category`는 `train_loader`를 요구, k-shot이 0이어도 `train_loader` 전체로 메모리 구축.
  - 준비 시간 기록을 위해 로깅 추가(선택).
- 추론 루프
  - `ForwardResult.pixel_scores`가 `(B, 1, H, W)` 형태일 때 기존 후처리(평균, concat) 그대로 사용 가능.
  - 필요 시 `model.config["score_map"]["resize_to"]` 정보를 사용해 `specify_resolution` 단계 이전에 torchvision resize 적용(선택).
- 평가 결과 저장
  - 기존 `summarize_metrics`, `save_metrics_csv`, `plot_summary.py`는 출력 포맷 유지하므로 변경 없음.
  - PatchCore 추가 메트릭(예: 메모리 크기, 구축시간)을 `ForwardResult.extra` 또는 `metrics` 딕셔너리에 확장하는 여지 마련.
- Latency 측정
  - `profile_model`이 호출할 때 PatchCore는 `predict`만 호출하면 되므로 기존 구조 재사용. 다만 PatchCore는 내부적으로 dataloader 전체를 순회하는 로직이 없도록 `forward`가 배치 단위 추론만 수행해야 함.

## 테스트 & 문서 업데이트 계획
- 기능 테스트
  - `configs/mvtec_patchcore.yaml`을 사용해 `python src/main.py --config configs/mvtec_patchcore.yaml --device cuda:0` 실행.
  - 우선 1~2개 카테고리(`bottle`, `cable`)에 대해 `data.loader.max_test_samples` 등을 활용해 소규모 검증.
  - 결과 디렉터리: `results/mvtec/patchcore/mvtec_patchcore/<timestamp>/` 생성 및 `metrics.csv`, `summary_metrics.csv` 확인.
- 회귀 확인
  - WinCLIP 구성(`configs/mvtec_winclip.yaml`) 재실행하여 기존 파이프라인이 영향을 받지 않았는지 비교.
  - 공통 유틸(`build_mvtec_dataloaders`) 수정 시 다른 메서드에도 동일 동작 보장 여부 체크.
- 문서화
  - `README.md`에 PatchCore 실행 방법, 신규 의존성 설치(`pip install faiss-cpu pretrainedmodels click`) 안내 추가.
  - `docs/patchcore_notes.md`를 `docs/patchcore_integration.md`로 승격하거나 README에서 링크.
- 자동화
  - 필요 시 `scripts/run_patchcore_mvtec.sh` 스크립트 추가해 다중 카테고리 일괄 실행 제공.

## 구현 경로 요약
- `src/methods/patchcore.py`: PatchCore 래퍼 및 레지스트리 등록.
- `configs/mvtec_patchcore.yaml`: MVTec-AD PatchCore 실험 설정.
- 사전 학습 사용 시 `model.pretrained` 블록을 활성화해 기존 체크포인트 로드.
  - 예: WR50 320px 실험 → `backbone: wideresnet50`, `input_size: [3, 320, 320]`, `transforms.resize: 366`, `transforms.center_crop: 320`, `pretrain_embed_dim: 1024`, `target_embed_dim: 1024`, `sampler: approx_greedy_coreset`, `nn.k: 1`, `pretrained.root: git_clone/patchcore-inspection/models/IM224_WR50_L2-3_P001_D1024-1024_PS-3_AN-1_S0`.
- `src/datasets/__init__.py`: YAML 기반 이미지/마스크 변환 파이프라인 자동 구성.
- `requirements.txt`: PatchCore 실행에 필요한 `click`, `faiss-cpu`, `pretrainedmodels` 추가.
- `README.md`: PatchCore 실행 절차 안내.

