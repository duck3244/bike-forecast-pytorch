# PyTorch 자전거 수요 예측 프로젝트

EFavDB의 bike-forecast 프로젝트를 PyTorch로 현대화한 딥러닝 기반 자전거 공유 수요 예측 시스템

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 프로젝트 소개

이 프로젝트는 딥러닝을 활용하여 자전거 공유 시스템의 시간별 수요를 예측합니다. 날씨 조건, 시간적 패턴, 기타 관련 특성들을 기반으로 정확한 대여 수요를 예측하여 자전거 공유 서비스의 효율적인 운영을 지원합니다.

### 원본 프로젝트와의 차이점

- ✅ PyTorch 기반으로 완전히 재작성
- ✅ MLP, LSTM 등 다양한 딥러닝 모델 지원
- ✅ 자동화된 특성 엔지니어링
- ✅ GPU 가속 지원
- ✅ 포괄적인 시각화 및 분석 도구
- ✅ 하이퍼파라미터 자동 튜닝
- ✅ 모델 비교 및 앙상블 기능

## ✨ 주요 기능

### 🔍 데이터 분석
- **탐색적 데이터 분석 (EDA)**: 시간적 패턴, 계절성, 날씨 영향 분석
- **자동 특성 엔지니어링**: 순환 인코딩, 상호작용 항, 파생 변수 생성
- **데이터 품질 검사**: 이상치 감지, 결측값 처리, 분포 분석
- **시각화**: 40개 이상의 자동 생성 차트 및 그래프

### 🤖 모델링
- **다양한 아키텍처**: MLP, LSTM, Residual Networks
- **scikit-learn 모델**: Linear Regression, Random Forest, Gradient Boosting, SVR
- **앙상블 방법**: 여러 모델의 조합으로 성능 향상
- **전이 학습**: 사전 학습된 가중치 활용 가능

### 🎛️ 최적화
- **하이퍼파라미터 튜닝**: Grid Search, Random Search, Bayesian Optimization
- **교차 검증**: K-Fold 교차 검증으로 일반화 성능 평가
- **학습 곡線 분석**: 데이터 크기에 따른 성능 변화 분석
- **조기 종료**: 과적합 방지를 위한 자동 학습 중단

### 📊 평가 및 보고
- **다양한 지표**: R², RMSE, MAE, MAPE
- **시각화**: 예측 vs 실제, 잔차 플롯, 학습 곡선
- **특성 중요도**: 모델 해석을 위한 특성 기여도 분석
- **자동 보고서**: Markdown 형식의 상세 성능 리포트

## 🛠️ 설치 방법

### 필수 요구사항

- Python 3.8 이상
- pip 또는 conda

### 방법 1: pip 설치 (권장)

```bash
# 가상환경 생성 (선택사항이지만 권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 방법 2: conda 설치

```bash
# Conda 환경 생성
conda create -n bike-forecast python=3.9
conda activate bike-forecast

# 의존성 설치
pip install -r requirements.txt
```

### 개발 환경 설정

```bash
# 개발 도구 포함 설치
pip install -r requirements.txt
pip install pytest black flake8 mypy jupyter

# 또는 Makefile 사용
make install-dev
```

## 🚀 빠른 시작

### 1. 가장 간단한 방법

```bash
# MLP 모델로 빠른 훈련 (합성 데이터 사용)
python main.py --model mlp --quick

# 결과 확인
ls outputs/
```

### 3. 단계별 실행

```bash
# 1단계: 데이터 탐색
python data_exploration.py

# 2단계: 모델 훈련
python model_training.py --models mlp

# 3단계: 모델 비교
python model_comparison.py

# 4단계: 하이퍼파라미터 튜닝
python hyperparameter_tuning.py
```

### 4. 자신의 데이터로 실행

```bash
# CSV 파일 준비 (필수 컬럼: datetime, season, weather, temp, humidity, windspeed, count)
python main.py --data your_data.csv --model mlp
```

## 📁 프로젝트 구조

```
bike-forecast-pytorch/
├── 📊 핵심 모듈
│   ├── bike_forecast_pytorch.py    # 메인 모델 구현 (MLP, LSTM, 데이터 처리)
│   ├── utils.py                    # 유틸리티 함수 (시각화, 평가, 설정)
│   └── main.py                     # 통합 실행 스크립트
│
├── 🔍 기능별 스크립트
│   ├── data_exploration.py         # 데이터 탐색 및 분석
│   ├── model_comparison.py         # 모델 아키텍처 비교
│   ├── hyperparameter_tuning.py    # 하이퍼파라미터 최적화
│   └── model_training.py           # 고급 훈련 기능
│
├── ⚙️ 설정 파일
│   ├── config.yaml                 # 프로젝트 설정
│   ├── requirements.txt            # Python 의존성
│   └── setup.py                    # 패키지 설정
│
├── 📝 문서
│   ├── README.md                   # 영문 문서
│
└── 📂 출력 디렉토리 (자동 생성)
    └── outputs/
        ├── models/                 # 저장된 모델 (.pth)
        ├── plots/                  # 시각화 결과 (.png)
        ├── predictions/            # 예측 결과 (.json)
        └── reports/                # 성능 리포트 (.md)
```

## 📖 사용 가이드

### 데이터 형식

입력 CSV 파일은 다음 컬럼들을 포함해야 합니다:

| 컬럼명 | 설명 | 타입 | 예시 |
|--------|------|------|------|
| `datetime` | 날짜 및 시간 | datetime | 2011-01-01 00:00:00 |
| `season` | 계절 (1-4) | int | 1=봄, 2=여름, 3=가을, 4=겨울 |
| `holiday` | 공휴일 여부 | int | 0=평일, 1=공휴일 |
| `workingday` | 근무일 여부 | int | 0=주말/휴일, 1=근무일 |
| `weather` | 날씨 상태 | int | 1=맑음, 2=안개, 3=비, 4=폭우 |
| `temp` | 기온 (°C) | float | 9.84 |
| `atemp` | 체감온도 (°C) | float | 14.395 |
| `humidity` | 습도 (%) | int | 81 |
| `windspeed` | 풍속 | float | 0.0 |
| `count` | 대여 수 (타겟) | int | 16 |

### 기본 사용법

#### 1. 데이터 탐색

```bash
# 기본 데이터 탐색
python data_exploration.py

# 커스텀 데이터로 탐색
python data_exploration.py --data my_bike_data.csv --output-dir outputs/my_analysis

# 플롯 생성 없이 빠른 분석
python data_exploration.py --no-plots
```

**생성되는 결과물:**
- 📊 시간대별/요일별/계절별 패턴 분석
- 🌤️ 날씨 영향 분석
- 📈 상관관계 히트맵
- 🎯 이상치 탐지 결과
- 📝 주요 인사이트 요약 리포트

#### 2. 모델 훈련

```bash
# 기본 MLP 훈련
python model_training.py --models mlp

# LSTM 모델 훈련
python model_training.py --models lstm

# 여러 모델 동시 훈련
python model_training.py --models mlp lstm

# 앙상블 모델 훈련
python model_training.py --models mlp lstm --ensemble

# 교차 검증 포함
python model_training.py --models mlp --cross-validation

# 학습 곡선 분석 포함
python model_training.py --models mlp --learning-curves
```

#### 3. 모델 비교

```bash
# 모든 모델 비교 (PyTorch + scikit-learn)
python model_comparison.py --epochs 30

# PyTorch 모델만 비교
python model_comparison.py --epochs 25 --no-sklearn

# 빠른 비교 (적은 에폭)
python model_comparison.py --epochs 15 --output-dir outputs/quick_comparison
```

#### 4. 하이퍼파라미터 튜닝

```bash
# Random Search (빠르고 효율적)
python hyperparameter_tuning.py --method random --max-combinations 20

# Grid Search (체계적 탐색)
python hyperparameter_tuning.py --method grid --max-combinations 15

# Bayesian Optimization (고급)
python hyperparameter_tuning.py --method bayesian --iterations 30

# LSTM 모델 튜닝
python hyperparameter_tuning.py --model lstm --method random
```

## 🏗️ 모델 아키텍처

### 1. MLP (Multi-Layer Perceptron)

**표준 MLP 구조:**
```
입력 (24 features) 
    ↓
[512] → BatchNorm → ReLU → Dropout(0.2)
    ↓
[256] → BatchNorm → ReLU → Dropout(0.2)
    ↓
[128] → BatchNorm → ReLU → Dropout(0.2)
    ↓
[64] → BatchNorm → ReLU → Dropout(0.2)
    ↓
[1] (출력)
```

**특징:**
- ✅ 빠른 훈련 속도
- ✅ 적은 메모리 사용
- ✅ 해석 가능성 높음
- ✅ 소규모 데이터셋에 적합

### 2. LSTM (Long Short-Term Memory)

**LSTM 구조:**
```
입력 (24 features)
    ↓
LSTM [128 hidden, 2 layers] → Dropout(0.2)
    ↓
FC [64] → ReLU → Dropout(0.2)
    ↓
FC [1] (출력)
```

**특징:**
- ✅ 시계열 패턴 학습
- ✅ 장기 의존성 포착
- ✅ 순차적 데이터에 최적화
- ⚠️ 훈련 시간이 상대적으로 김

### 3. Residual MLP

**잔차 연결이 있는 심층 네트워크:**
```
입력 → [256]
    ↓
[Residual Block 1] (256→256)
    ↓
[Residual Block 2] (256→128)
    ↓
FC [1] (출력)
```

**특징:**
- ✅ 깊은 네트워크 훈련 가능
- ✅ 그래디언트 소실 방지
- ✅ 복잡한 패턴 학습
- ⚠️ 더 많은 파라미터 필요

### 특성 엔지니어링

자동으로 생성되는 특성들:

1. **시간 특성**
   - 시간, 일, 월, 연도, 요일

2. **순환 인코딩** (Cyclical Encoding)
   - `hour_sin`, `hour_cos`
   - `month_sin`, `month_cos`
   - `weekday_sin`, `weekday_cos`

3. **파생 특성**
   - `is_rush_hour`: 출퇴근 시간 (7-9시, 17-19시)
   - `is_weekend`: 주말 여부
   - `temp_humidity`: 온도×습도 상호작용
   - `temp_windspeed`: 온도×풍속 상호작용

## 📈 성능 평가

### 평가 지표

| 지표 | 설명 | 좋은 값 |
|------|------|---------|
| **R²** (결정계수) | 모델이 설명하는 분산 비율 | > 0.85 |
| **RMSE** | 평균 제곱근 오차 | < 20 |
| **MAE** | 평균 절대 오차 | < 15 |
| **MAPE** | 평균 절대 백분율 오차 | < 15% |

### 예상 성능 (합성 데이터 기준)

| 모델 | R² | RMSE | MAE | 훈련 시간 |
|------|-----|------|-----|----------|
| Simple MLP | 0.87 | 18.2 | 13.5 | ~15초 |
| Standard MLP | 0.92 | 14.8 | 10.2 | ~25초 |
| LSTM | 0.90 | 16.1 | 11.8 | ~45초 |
| Random Forest | 0.89 | 17.3 | 12.4 | ~8초 |
| Gradient Boosting | 0.91 | 15.5 | 11.0 | ~20초 |
| Ensemble | 0.93 | 13.9 | 9.8 | N/A |

### 성능 해석

**우수한 성능 (Excellent)**
- R² > 0.9, RMSE < 10
- 프로덕션 배포 가능

**좋은 성능 (Good)**
- R² > 0.8, RMSE < 20
- 실용적 사용 가능

**보통 성능 (Fair)**
- R² > 0.7, RMSE < 30
- 추가 개선 필요

**낮은 성능 (Poor)**
- R² < 0.7, RMSE > 30
- 모델 재검토 필요

## 🐛 문제 해결

### 일반적인 문제들

#### 1. CUDA 메모리 부족

```bash
# 에러: RuntimeError: CUDA out of memory

# 해결 방법 1: 배치 크기 줄이기
python main.py --model mlp  # config.yaml에서 batch_size 감소

# 해결 방법 2: CPU 사용
python main.py --model mlp --device cpu
```

#### 2. 모듈을 찾을 수 없음

```bash
# 에러: ModuleNotFoundError: No module named 'torch'

# 해결 방법: 의존성 재설치
pip install -r requirements.txt

# 또는 가상환경 확인
which python  # 올바른 Python 인터프리터 사용 중인지 확인
```

#### 3. 훈련이 너무 느림

```bash
# 해결 방법 1: GPU 사용 확인
python -c "import torch; print(torch.cuda.is_available())"

# 해결 방법 2: 빠른 모드 사용
python main.py --model mlp --quick

# 해결 방법 3: 데이터 크기 줄이기
# config.yaml에서 synthetic_samples 감소
```

#### 4. 성능이 낮음

```bash
# 진단 1: 데이터 품질 확인
python data_exploration.py --data your_data.csv

# 진단 2: 학습 곡선 확인
python model_training.py --models mlp --learning-curves

# 해결책: 하이퍼파라미터 튜닝
python hyperparameter_tuning.py --method random
```

### 디버깅 팁

```bash
# 상세 로그 활성화
export PYTHONUNBUFFERED=1

# 단계별 실행으로 디버깅
python -m pdb main.py --model mlp

# 특정 단계만 실행
python data_exploration.py  # 데이터만 확인
python model_training.py --models mlp --output-dir debug  # 훈련만
```

## 🔧 설정 파일 (config.yaml)

```yaml
# 데이터 설정
data:
  synthetic_samples: 8760  # 생성할 샘플 수 (1년 = 24*365)
  test_size: 0.2           # 테스트 세트 비율
  val_size: 0.2            # 검증 세트 비율
  random_seed: 42          # 재현성을 위한 시드
  target_column: "count"   # 타겟 변수명

# 훈련 설정
training:
  epochs: 100              # 훈련 에폭 수
  batch_size: 64           # 배치 크기
  learning_rate: 0.001     # 학습률
  weight_decay: 1e-5       # L2 정규화
  early_stopping_patience: 20  # 조기 종료 인내심

# 모델 설정
models:
  mlp:
    hidden_sizes: [512, 256, 128, 64]  # 은닉층 크기
    dropout_rate: 0.2                   # 드롭아웃 비율
  lstm:
    hidden_size: 128         # LSTM 은닉 크기
    num_layers: 2            # LSTM 레이어 수
    dropout: 0.2             # 드롭아웃 비율

# 출력 설정
output:
  model_save_path: "models/"
  save_predictions: true
  plot_results: true
```

---