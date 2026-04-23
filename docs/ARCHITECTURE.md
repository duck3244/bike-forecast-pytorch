# 아키텍처 문서

PyTorch 기반 자전거 수요 예측 시스템의 전체 아키텍처를 설명합니다.

## 1. 시스템 개요

본 프로젝트는 **PyTorch 기반 딥러닝 모델(MLP/LSTM)** 을 이용해 자전거 공유 수요를 예측하는 풀스택 애플리케이션입니다.

- **Backend**: FastAPI + PyTorch (REST API, 학습 작업 관리, SSE 스트리밍)
- **Frontend**: React + TypeScript + Vite + Tailwind CSS (대시보드, 학습 모니터링, 예측 폼)
- **통신**: REST API + Server-Sent Events(SSE, 실시간 학습 진행 상황 스트림)

## 2. 디렉터리 구조

```
bike-forecast-pytorch/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI 앱, 라우트, 상태 관리
│   │   ├── routers/
│   │   │   ├── datasets.py          # CSV 업로드, 데이터셋 메타데이터
│   │   │   └── training.py          # 학습 작업 API, SSE 스트리밍
│   │   └── services/
│   │       └── training.py          # 백그라운드 학습 실행, 작업 레지스트리
│   ├── bike_forecast_pytorch.py     # 핵심 모델/학습기 (BikeForecaster, LSTMForecaster)
│   ├── utils.py                     # 설정, 모델 저장/로드, 지표, 시각화
│   ├── data_exploration.py          # EDA 스크립트 (오프라인)
│   ├── model_comparison.py          # 모델 비교 스크립트 (오프라인)
│   ├── model_training.py            # 학습 스크립트 (오프라인)
│   ├── hyperparameter_tuning.py     # 하이퍼파라미터 탐색 (오프라인)
│   ├── main.py                      # CLI 엔트리포인트
│   ├── config.yaml                  # 기본 설정 (데이터/모델/학습)
│   └── data/train.csv               # 기본 학습 데이터셋
├── frontend/
│   ├── src/
│   │   ├── main.tsx                 # React 엔트리, QueryClient, Router
│   │   ├── api/
│   │   │   ├── client.ts            # fetch 래퍼 + 타입드 API 메서드
│   │   │   └── schema.ts            # OpenAPI 자동 생성 타입
│   │   ├── components/{AppShell,Card}.tsx
│   │   ├── pages/{HomePage,PredictPage,TrainingPage}.tsx
│   │   └── hooks/useTrainingStream.ts  # SSE EventSource 훅
│   ├── package.json                 # React 18, TanStack Query, Recharts
│   └── vite.config.ts
└── docs/
    ├── ARCHITECTURE.md
    └── UML.md
```

## 3. 전체 아키텍처 다이어그램

```
┌──────────────────────────────────────────────────────────────┐
│                    브라우저 (Client)                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  React SPA (Vite, TypeScript, Tailwind)                │  │
│  │  ┌─────────────┐ ┌──────────────┐ ┌────────────────┐   │  │
│  │  │  HomePage   │ │ PredictPage  │ │ TrainingPage   │   │  │
│  │  └─────────────┘ └──────────────┘ └────────────────┘   │  │
│  │         ▲               ▲                ▲              │  │
│  │         │   TanStack Query (캐시/폴링)    │              │  │
│  │  ┌──────┴───────────────┴────────────────┴───────────┐  │  │
│  │  │   api/client.ts  (REST)   useTrainingStream (SSE)│  │  │
│  │  └───────────────────────┬──────────────────────────┘  │  │
│  └──────────────────────────┼──────────────────────────────┘  │
└─────────────────────────────┼──────────────────────────────────┘
                              │ HTTP / SSE
┌─────────────────────────────▼──────────────────────────────────┐
│              FastAPI Server (Uvicorn, ASGI)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  app/main.py  ── CORS, lifespan, 전역 STATE + Lock        │  │
│  │   ├─ /api/health, /api/models/current/metrics             │  │
│  │   ├─ /api/predict                                         │  │
│  │   ├─ routers/datasets.py  (업로드/목록/요약)              │  │
│  │   └─ routers/training.py  (작업 CRUD + SSE events)        │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  services/training.py                                     │  │
│  │   ├─ TrainingJob (dataclass, 상태/이벤트 큐)               │  │
│  │   ├─ _REGISTRY  (in-memory, Lock 보호)                     │  │
│  │   └─ _run_training  (daemon thread)                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Core ML  (bike_forecast_pytorch.py / utils.py)           │  │
│  │   ├─ BikeDataProcessor  (피처 엔지니어링, 스케일러)         │  │
│  │   ├─ BikeDataset (torch.utils.data.Dataset)               │  │
│  │   ├─ BikeForecaster (MLP), LSTMForecaster                 │  │
│  │   └─ BikeForecasterTrainer (train/predict/AMP/early stop) │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────┬─────────────────────────────────────────┘
                       │ 파일 I/O
┌──────────────────────▼─────────────────────────────────────────┐
│                     로컬 파일 시스템                           │
│  backend/data/train.csv       (기본 학습 데이터)                │
│  backend/outputs/uploads/     (사용자 업로드 CSV)               │
│  backend/outputs/models/                                        │
│    ├─ {model_type}_model.pth  (state_dict)                     │
│    └─ preprocessor.joblib     (scalers, feature_cols, meta)    │
│  backend/outputs/predictions/ (예측 결과 JSON)                  │
│  backend/outputs/plots/       (학습 곡선, 잔차 플롯)             │
└────────────────────────────────────────────────────────────────┘
```

## 4. Backend 구성요소

### 4.1 FastAPI 애플리케이션 (`backend/app/main.py`)

- `FastAPI(title="Bike Demand Forecast API", version="0.2.0", lifespan=lifespan)`
- **CORS**: `http://localhost:5173` (Vite 개발 서버) 허용
- **전역 상태**:
  - `STATE: dict` — 현재 로드된 모델/프로세서/피처 컬럼/메타데이터 캐시
  - `STATE_LOCK: threading.Lock` — 다중 요청과 학습 스레드 간 직렬화
- **lifespan**: 앱 시작 시 `reload_inference_model()`을 호출해 최신 체크포인트를 메모리에 로드

### 4.2 라우터

| 라우터 | 경로 | 설명 |
|---|---|---|
| `app/main.py` | `GET /api/health` | 모델 로드 상태, 모델 클래스, 입력 크기 |
| `app/main.py` | `GET /api/models/current/metrics` | 최근 학습 결과 지표 (R², RMSE, MAE, MSE, MAPE) |
| `app/main.py` | `POST /api/predict` | 단일 샘플 예측 |
| `routers/datasets.py` | `GET /api/datasets/summary` | 활성 데이터셋 요약 (행 수, 기간, 타깃 통계) |
| `routers/datasets.py` | `GET /api/datasets` | 업로드된 CSV 목록 |
| `routers/datasets.py` | `POST /api/datasets/upload` | multipart CSV 업로드 + 스키마 검증 |
| `routers/training.py` | `POST /api/training/jobs` | 학습 작업 생성 및 시작 (202 Accepted) |
| `routers/training.py` | `GET /api/training/jobs` | 작업 목록 |
| `routers/training.py` | `GET /api/training/jobs/{id}` | 단일 작업 상태 |
| `routers/training.py` | `DELETE /api/training/jobs/{id}` | 학습 취소 (idempotent) |
| `routers/training.py` | `GET /api/training/jobs/{id}/events` | SSE 스트림 (epoch/done/cancelled/error) |

### 4.3 학습 서비스 (`backend/app/services/training.py`)

- **`JobStatus` (Enum)**: `PENDING → RUNNING → COMPLETED | CANCELLED | FAILED`
- **`TrainingJob` (dataclass)**: id, 설정, 상태, 타임스탬프, 지표, 에러, epoch 이력, 내부 큐/취소 이벤트
- **`_REGISTRY`**: 인메모리 dict + Lock (프로세스 재시작 시 초기화됨)
- **실행 모델**:
  - `start_job()` → daemon thread로 `_run_training()` 실행
  - 학습 루프에서 `on_epoch_end` 콜백이 이벤트를 `asyncio.Queue`에 넣음
  - SSE 핸들러(`drain_events`)가 큐에서 꺼내 클라이언트로 전송
  - `threading.Event` 기반 협조적 취소 (`should_stop()` 폴링)
- **완료 시 콜백**: `on_completion` → `reload_inference_model()`로 STATE 원자적 교체

### 4.4 핵심 ML 모듈 (`backend/bike_forecast_pytorch.py`)

| 클래스 | 역할 |
|---|---|
| `BikeDataset(torch.utils.data.Dataset)` | numpy 배열을 float32 텐서로 래핑 |
| `BikeForecaster(nn.Module)` | MLP: Linear → BatchNorm → ReLU → Dropout 반복 |
| `LSTMForecaster(nn.Module)` | LSTM(2층, hidden=128) + FC 헤드 |
| `BikeDataProcessor` | 합성 데이터 생성, 피처 엔지니어링(24개), StandardScaler 적합 |
| `BikeForecasterTrainer` | 학습 루프, AMP, `ReduceLROnPlateau`, Early Stopping(patience=20), epoch 콜백, 취소 폴링 |

### 4.5 유틸리티 (`backend/utils.py`)

- **설정**: `load_config(path)`, `get_default_config()`
- **영속화**: `save_model(model, path, metadata, processor, feature_cols)` — `.pth` + `preprocessor.joblib` 사이드카로 원자적 저장
- **로딩**: `load_checkpoint(model_dir, model_class_map)` — `weights_only=True`로 안전 로드
- **지표**: `calculate_metrics` → `{mse, rmse, mae, r2, mape}`
- **데이터 분할**: `time_series_split(X, y, test_size, val_size)` — 시계열 순서 보존(셔플 없음)

### 4.6 설정 (`backend/config.yaml`)

`data`, `features`, `models` (mlp / lstm), `training` (epochs, batch_size, lr, early_stopping), `device`, `output` 섹션으로 구성. 학습 API 요청에서 개별 필드를 덮어쓸 수 있습니다.

## 5. Frontend 구성요소

### 5.1 스택

- **React 18 + TypeScript**, Vite 5 빌드
- **TanStack React Query v5**: `staleTime=30s`, focus refetch 비활성
- **React Router v6**: `/`, `/train`, `/predict`
- **Recharts 3**: 학습 곡선(train/val loss) 라인 차트
- **Tailwind CSS 3**: 유틸리티 기반 스타일

### 5.2 페이지

| 페이지 | 경로 | 설명 |
|---|---|---|
| `HomePage` | `/` | 헬스, 현재 모델 지표, 데이터셋 요약 카드 |
| `PredictPage` | `/predict` | 단일 예측 폼 (datetime, 계절, 날씨, 기온, 습도 등) |
| `TrainingPage` | `/train` | 데이터셋 업로드, 학습 파라미터 설정, 작업 시작/취소, SSE 기반 실시간 학습 곡선 |

### 5.3 API 클라이언트 (`frontend/src/api/client.ts`)

- `request<T>(path, init?)`: JSON 응답 파싱, 에러 본문 포함 throw
- `FormData`는 Content-Type 자동 지정 회피
- 메서드: `health`, `metrics`, `datasetSummary`, `predict`, `listDatasets`, `uploadDataset`, `listJobs`, `getJob`, `startJob`, `cancelJob`

### 5.4 실시간 스트림 (`frontend/src/hooks/useTrainingStream.ts`)

- `EventSource('/api/training/jobs/{id}/events')` 연결
- 이벤트별 핸들러:
  - `epoch` → `epochs[]`에 누적 (Recharts 데이터 소스)
  - `done` → `status='completed'`, 최종 metrics 저장
  - `cancelled` → `status='cancelled'`
  - `error` → `status='failed'`, error 메시지 저장
- `jobId` 변경 시 기존 연결 close, 새 연결 수립

## 6. 주요 데이터 플로우

### 6.1 학습 플로우

1. **업로드**: 사용자가 CSV 업로드 → `/api/datasets/upload` → 스키마 검증 후 `outputs/uploads/{id}.csv` 저장
2. **작업 생성**: `POST /api/training/jobs` (model_type, epochs, batch_size, learning_rate, dataset_id) → `register_job` → 202 반환
3. **스레드 실행**: `start_job` → daemon thread에서 `_run_training`
   - CSV 로드 → `BikeDataProcessor.prepare_data()` (피처 엔지니어링 + 스케일링 + time-series split)
   - `BikeForecasterTrainer.build_model()` → `train()` 루프
   - 매 epoch마다 `on_epoch_end`가 이벤트를 큐에 푸시
4. **SSE 수신**: 프론트가 `GET .../events` 구독 → 학습 곡선 실시간 업데이트
5. **완료**: 테스트셋 예측 → 지표 계산 → `save_model()` → `on_completion`이 `reload_inference_model()`로 STATE 교체 → `done` 이벤트 방출

### 6.2 예측 플로우

1. 사용자가 폼 제출 → `POST /api/predict`
2. 서버가 `STATE_LOCK` 하에서 모델/프로세서/피처 컬럼 조회
3. 단일 행 DataFrame 생성 → `engineer_features()` → 저장된 StandardScaler로 transform
4. LSTM인 경우 `(1, 1, 24)`, MLP인 경우 `(1, 24)` 텐서로 변환 후 `torch.no_grad()` 추론
5. 음수 클램핑 후 `PredictResponse(count, count_rounded)` 반환

## 7. 모델 영속화 포맷

- **체크포인트 `.pth`**: PyTorch `state_dict` (가중치만, `weights_only=True` 로드)
- **사이드카 `preprocessor.joblib`**:
  - `model_class` (`"BikeForecaster"` | `"LSTMForecaster"`)
  - `input_size` (기본 24)
  - `feature_cols` (순서 보존)
  - `scalers["numerical"]` (fitted `StandardScaler`)
  - `metadata` (지표, 학습 시간, job_id 등)

이 분리로 **가중치 로딩은 안전(`weights_only=True`)** 하게 유지하면서 전처리기의 재현성을 보장합니다.

## 8. 동시성 및 상태 관리

| 자원 | 보호 메커니즘 |
|---|---|
| `STATE` (추론 모델) | `STATE_LOCK` (threading.Lock) — 추론 요청과 모델 재로딩 직렬화 |
| `_REGISTRY` (작업 목록) | `_REGISTRY_LOCK` — 작업 등록/조회 보호 |
| 학습 취소 | `threading.Event` — 학습 루프에서 `should_stop()` 폴링 |
| 이벤트 전송 | `asyncio.Queue` + `asyncio.EventLoop` 참조 — 스레드 → async 브리지 |

## 9. 외부 의존성 요약

**Backend**: `torch`, `numpy`, `pandas`, `scikit-learn`, `fastapi`, `uvicorn`, `python-multipart`, `sse-starlette`, `pyyaml`, `joblib`, `matplotlib`, `seaborn`, `plotly`, `tqdm`

**Frontend**: `react`, `react-dom`, `react-router-dom`, `@tanstack/react-query`, `recharts`, `tailwindcss`, `vite`, `typescript`

## 10. 확장 포인트

- **새 모델 추가**: `bike_forecast_pytorch.py`에 `nn.Module` 서브클래스 추가 → `BikeForecasterTrainer.build_model()`에 분기 → `utils.load_checkpoint`의 `model_class_map` 등록
- **영속 저장소**: 현재 `_REGISTRY`가 인메모리이므로, 재시작 시 작업 이력이 사라집니다. SQLite/Redis로 교체 가능
- **다중 모델 서빙**: STATE를 단일 슬롯 → `{model_id: STATE}` 맵으로 확장하면 A/B 테스트 가능
- **인증/인가**: 현재 open API — 프로덕션 배포 시 FastAPI `Depends`로 인증 미들웨어 추가 필요
