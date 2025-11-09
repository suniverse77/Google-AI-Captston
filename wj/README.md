# WinCLIP Baseline Scaffold

이 저장소는 WinCLIP 기반의 Zero/Few-shot Anomaly Detection 실험 파이프라인을 TinyCLIP 등 경량 모델로 확장하기 쉽게 구성한 스캐폴드입니다.

## 디렉터리 구조
- `configs/` : 실험 설정 YAML (공통 설정, 모델별 세부 설정)
- `data/` : MVTec-AD 등 데이터셋 위치 (심볼릭 링크 또는 다운로드 스크립트 결과)
- `src/` : 메인 코드
  - `main.py` : 실험 진입점 (config 로드 → 데이터셋 → 모델 → 평가/시각화)
  - `datasets/` : 데이터셋 로더 및 전처리
  - `methods/` : WinCLIP/TinyCLIP 등 모델 구현
  - `eval/` : 정량 평가 및 효율성 측정 모듈
  - `vis/` : 히트맵 overlay 등 시각화
  - `utils/` : 설정/로그/시드/타이머 등 공통 유틸
- `scripts/` : 대량 실험 실행, 결과 취합, 시각화 스크립트
- `results/` : 실험 산출물 (metrics.csv, summary.csv, 그래프 등)

## WinCLIP 코드 의존성
원본 WinCLIP 구현을 `src/methods/winclip/original/`에 벤더링해 두었으므로 추가 외부 레포 의존성 없이 바로 사용할 수 있습니다.

## 새로운 데이터셋 추가 방법
1. **데이터셋 정보 작성**  
   - `configs/` 아래에 새로운 YAML을 만들고 `data.dataset`에 데이터셋 이름을 명시합니다.  
   - 카테고리 목록, 루트 경로, k-shot 등의 옵션을 설정합니다.
2. **Dataset 클래스 구현**  
   - `src/datasets/`에 `<dataset>.py` 파일을 추가하고 `BaseDataset`을 상속받는 클래스를 정의합니다.  
   - `loader`에서 사용할 수 있도록 `create_<dataset>_dataset`, `build_<dataset>_dataloaders` 보조 함수를 함께 작성합니다.
3. **등록 및 import**  
   - `src/datasets/__init__.py`에 helper 함수를 등록하고, `src/main.py`에서 조건에 맞게 호출하거나 새로운 실행 스크립트를 작성합니다.
4. **실행**  
   - `python src/main.py --config configs/<dataset>_<model>.yaml`처럼 새로운 설정 파일을 지정해 평가를 수행합니다.

## 추가 모델 통합 방법
1. **모델 코드 준비**  
   - 새 모델을 `src/methods/<model_name>/` 이하에 두거나, 벤더링이 필요하면 서브 폴더를 만들어 정리합니다.
   - 원본 코드가 필요한 경우 `__init__.py`를 만들어 노출할 클래스를 지정합니다.
2. **Method 클래스 구현**  
   - `src/methods/`에 `<model_name>.py` 파일을 추가하고 `BaseMethod`를 상속하는 클래스를 작성합니다.  
   - `@register_method("<model_name>")`를 붙여 레지스트리에 등록하고, `from_config`, `setup`, `forward`를 구현합니다.
3. **환경 설정**  
   - 필요한 패키지가 있으면 `requirements.txt`에 추가합니다.  
   - `configs/`에서 `model.name`을 새 모델 이름으로 설정하고, 필요한 하이퍼 파라미터를 넘기도록 합니다.
4. **실행**  
   - `python src/main.py --config configs/<dataset>_<model>.yaml`로 새 모델을 평가합니다.  
   - 결과 구조는 기존 WinCLIP과 동일하게 `results/` 아래에 저장됩니다.

## 추후 작업 예정
- TinyCLIP 기반 방법 추가
- 다양한 데이터셋 구성 및 자동 다운로드 스크립트
- CI 기반 결과 검증 및 리포트 자동화