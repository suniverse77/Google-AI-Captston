# Anomaly Detection Framework

Zero-shot 및 Few-shot 이상 탐지(Anomaly Detection)를 위한 통합 실험 프레임워크입니다. WinCLIP, PatchCore, AF-CLIP 등 다양한 방법론을 지원하며, 경량 CLIP 모델(TinyCLIP 등) 확장이 용이하도록 설계되었습니다.

## 주요 기능

- **다양한 이상 탐지 방법 지원**
  - WinCLIP: Zero-shot 이상 탐지 방법론
  - PatchCore: Memory bank 기반 방법론
  - AF-CLIP: 경량 CLIP 모델 기반 방법론
  - miniCLIP: 더욱 경량화된 버전

- **다양한 데이터셋 지원**
  - MVTec-AD: 산업 제품 이상 탐지 벤치마크
  - VisA: 다양한 물체의 이상 탐지 데이터셋

- **포괄적인 평가 지표**
  - Image-level: AUROC, F1 Score
  - Pixel-level: AUROC, F1 Score, PRO (Per-Region Overlap)
  - 성능 측정: Latency, Throughput, GPU 메모리 사용량
  - 모델 복잡도: 파라미터 수, FLOPs

- **시각화 도구**
  - 히트맵 overlay
  - 결과 비교 및 시각화

## 설치

### 요구사항

- Python >= 3.8
- PyTorch >= 2.0
- CUDA (GPU 사용 시)

### 의존성 설치

```bash
pip install -r requirements.txt
```

주요 의존성:
- `torch`, `torchvision`: PyTorch 프레임워크
- `open_clip-torch`: CLIP 모델 지원
- `faiss-cpu`: PatchCore의 메모리 뱅크 검색
- `scikit-learn`, `scikit-image`: 평가 및 전처리
- `opencv-python`, `Pillow`: 이미지 처리
- `timm`: Vision Transformer 모델 지원

## 빠른 시작

### 1. 데이터셋 준비

#### MVTec-AD
```bash
# MVTec-AD 데이터셋을 data/mvtec/ 디렉토리에 압축 해제
cd data/mvtec
# 다운로드한 데이터셋 압축 해제
tar -xf mvtec_anomaly_detection.tar.xz
```

#### VisA
```bash
# VisA 데이터셋을 data/visa/ 디렉토리에 압축 해제
cd data/visa
# 다운로드한 데이터셋 압축 해제
tar -xf VisA_20220922.tar
```

### 2. WinCLIP 실행 예제

```bash
# MVTec-AD 데이터셋에 대해 WinCLIP 실행
python src/main.py --config configs/mvtec_winclip.yaml --device cuda:0

# VisA 데이터셋에 대해 WinCLIP 실행
python src/main.py --config configs/visa_winclip.yaml --device cuda:0
```

### 3. PatchCore 실행 예제

```bash
# MVTec-AD 데이터셋에 대해 PatchCore 실행
python src/main.py --config configs/mvtec_patchcore.yaml --device cuda:0

# VisA 데이터셋에 대해 PatchCore 실행
python src/main.py --config configs/visa_patchcore.yaml --device cuda:0
```

### 4. AF-CLIP 실행 예제

```bash
# AF-CLIP 실행 (MVTec-AD)
python src/main.py --config configs/mvtec_afclip.yaml --device cuda:0

# miniCLIP 실행 (MVTec-AD)
python src/main.py --config configs/mvtec_afclip_tiny.yaml --device cuda:0
```

## 프로젝트 구조

```
.
├── configs/                 # 실험 설정 YAML 파일
│   ├── base.yaml           # 공통 기본 설정
│   ├── mvtec_winclip.yaml  # MVTec WinCLIP 설정
│   ├── visa_winclip.yaml   # VisA WinCLIP 설정
│   ├── mvtec_patchcore.yaml
│   └── ...
├── data/                   # 데이터셋 디렉토리
│   ├── mvtec/             # MVTec-AD 데이터
│   └── visa/              # VisA 데이터
├── src/                    # 메인 소스 코드
│   ├── main.py            # 실험 진입점
│   ├── datasets/          # 데이터셋 로더
│   │   ├── mvtec.py      # MVTec 데이터셋
│   │   ├── visa.py       # VisA 데이터셋
│   │   └── base.py       # 기본 데이터셋 클래스
│   ├── methods/           # 이상 탐지 방법 구현
│   │   ├── winclip.py    # WinCLIP 구현
│   │   ├── patchcore.py  # PatchCore 구현
│   │   ├── afclip.py     # AF-CLIP 구현
│   │   ├── afclip_tiny.py # miniCLIP 구현
│   │   └── base.py       # 기본 메서드 인터페이스
│   ├── eval/              # 평가 모듈
│   │   ├── metrics.py    # 메트릭 계산
│   │   ├── latency.py    # 성능 측정
│   │   └── utils.py      # 평가 유틸리티
│   ├── utils/             # 공통 유틸리티
│   │   ├── config.py     # 설정 로더
│   │   ├── logger.py     # 로깅
│   │   └── seed.py       # 시드 설정
│   └── vis/               # 시각화
│       └── overlay.py    # 히트맵 overlay
├── scripts/               # 스크립트
│   ├── eval_collect.py   # 결과 수집
│   └── plot_summary.py   # 결과 시각화
├── results/               # 실험 결과 저장
│   ├── mvtec/
│   └── visa/
├── train_afclip.py        # AF-CLIP 학습 스크립트
└── requirements.txt       # 의존성 목록
```

## 설정 파일 구조

설정 파일은 YAML 형식으로 작성되며, 다음과 같은 구조를 가집니다:

```yaml
experiment:
  name: experiment_name
  seed: 2025
  output_root: results/mvtec/winclip

data:
  dataset: mvtec
  root: data/mvtec
  categories: ["bottle", "cable", ...]
  batch_size: 4
  k_shot: 0  # Zero-shot 설정

model:
  name: winclip
  device: cuda
  backbone: ViT-B-16-plus-240
  resolution: 400

evaluation:
  metrics:
    compute_image_auroc: true
    compute_pixel_auroc: true
    compute_pro: true
  latency:
    warmup_iters: 1000
    measure_iters: 1000

visualization:
  num_examples: 12
  save_overlay: true
```

## 결과 해석

실험 실행 후 결과는 `results/<dataset>/<method>/<experiment_name>/` 디렉토리에 저장됩니다:

- `summary_metrics.csv`: 전체 카테고리 요약 메트릭
- `summary_metrics_pretty.csv`: 가독성 향상된 메트릭
- `summary_metrics_pretty.md`: 마크다운 형식 메트릭
- `<category>/metrics.csv`: 카테고리별 상세 메트릭
- `<category>/overlays/`: 시각화 결과 이미지

### 주요 메트릭

- **Image AUROC**: 이미지 레벨 이상 탐지 성능
- **Pixel AUROC**: 픽셀 레벨 이상 탐지 성능
- **PRO AUROC**: Per-Region Overlap 성능
- **F1 Score**: 정밀도와 재현율의 조화 평균
- **Latency**: 추론 지연 시간 (ms)
- **Throughput (FPS)**: 초당 처리 프레임 수
- **VRAM Peak**: 최대 GPU 메모리 사용량

## 새로운 데이터셋 추가

1. **데이터셋 로더 구현**
   - `src/datasets/<dataset>.py` 파일 생성
   - `BaseDataset` 클래스 상속
   - `create_<dataset>_dataset` 및 `build_<dataset>_dataloaders` 함수 구현

2. **설정 파일 작성**
   - `configs/<dataset>_<method>.yaml` 생성
   - 데이터셋 경로 및 카테고리 설정

3. **메인 파일에 등록**
   - `src/main.py`의 `DATASET_BUILDERS` 딕셔너리에 추가

## 새로운 방법론 추가

1. **메서드 클래스 구현**
   - `src/methods/<method>.py` 파일 생성
   - `BaseMethod` 클래스 상속
   - `@register_method("<method_name>")` 데코레이터 추가
   - `from_config`, `setup`, `forward` 메서드 구현

2. **설정 파일 작성**
   - `configs/<dataset>_<method>.yaml` 생성
   - 모델별 하이퍼파라미터 설정

3. **메인 파일에 import**
   - `src/main.py`에서 메서드 모듈 import

## AF-CLIP 학습

AF-CLIP 모델을 학습하려면:

```bash
python src/train_afclip.py --config configs/train_afclip.yaml --device cuda:0
```

학습된 모델은 `results/` 디렉토리에 저장되며, 평가 시 설정 파일에서 체크포인트 경로를 지정할 수 있습니다.

## 성능 측정

프레임워크는 다음과 같은 성능 지표를 자동으로 측정합니다:

- **Latency**: 배치 크기 1에서의 단일 추론 지연 시간
- **Throughput**: 배치 크기 16에서의 초당 처리량
- **GPU 메모리**: 추론 시 최대 VRAM 사용량
- **모델 복잡도**: 총 파라미터 수, 학습 가능 파라미터 수, FLOPs

## 시각화

설정 파일에서 `visualization.save_overlay: true`로 설정하면, 각 카테고리별로 히트맵 overlay 이미지가 생성됩니다:

```bash
# 결과 확인
ls results/mvtec/winclip/<experiment_name>/<category>/overlays/
```

## 주요 특징

- **모듈화된 구조**: 각 방법론과 데이터셋이 독립적으로 구현되어 쉽게 확장 가능
- **통합 평가 파이프라인**: 모든 방법론에 대해 동일한 평가 메트릭 사용
- **자동화된 결과 관리**: 실험별로 자동 디렉토리 생성 및 결과 저장
- **설정 기반 실행**: YAML 파일로 모든 하이퍼파라미터 관리
- **재현 가능성**: 시드 설정 및 결정적 실행 지원

## 라이선스

프로젝트에 사용된 각 방법론의 원본 라이선스를 확인하세요:
- WinCLIP: 원본 WinCLIP 저장소의 라이선스 참조
- PatchCore: 원본 PatchCore 저장소의 라이선스 참조
- TinyCLIP: 원본 TinyCLIP 저장소의 라이선스 참조

## 참고 자료

- [WinCLIP](https://github.com/amazon-science/winclip)
- [PatchCore](https://github.com/amazon-science/patchcore-inspection)
- [TinyCLIP](https://github.com/microsoft/UniCL)
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA Dataset](https://github.com/amazon-science/spot-the-diff)

## 기여

이슈 리포트와 풀 리퀘스트를 환영합니다. 기여 전에 먼저 이슈를 열어 작업 내용을 논의해 주세요.
