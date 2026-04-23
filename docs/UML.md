# UML 다이어그램

Mermaid 기반 UML 다이어그램 모음입니다. GitHub/GitLab/VSCode Markdown 프리뷰에서 바로 렌더링됩니다.

## 1. 클래스 다이어그램 — Backend 핵심

```mermaid
classDiagram
    class BikeDataset {
        -Tensor features
        -Tensor targets
        +__init__(features, targets)
        +__len__() int
        +__getitem__(idx) tuple
    }

    class BikeForecaster {
        -nn.Sequential network
        +__init__(input_size, hidden_sizes, dropout_rate)
        +forward(x) Tensor
    }

    class LSTMForecaster {
        -nn.LSTM lstm
        -nn.Sequential fc
        +__init__(input_size, hidden_size, num_layers, dropout)
        +forward(x) Tensor
    }

    class BikeDataProcessor {
        -Dict~str,StandardScaler~ scalers
        -Dict encoders
        +create_sample_data(n_samples) DataFrame
        +engineer_features(df) DataFrame
        +prepare_data(df, target_col) tuple
    }

    class BikeForecasterTrainer {
        -torch.device device
        -str model_type
        -nn.Module model
        -Dict history
        +build_model(input_size) nn.Module
        +train(X_train, y_train, X_val, y_val, epochs, batch_size, lr, on_epoch_end, should_stop) Dict
        +predict(X_test) ndarray
        +plot_training_history()
    }

    class TorchDataset {
        <<torch.utils.data.Dataset>>
    }

    class NnModule {
        <<nn.Module>>
    }

    TorchDataset <|-- BikeDataset
    NnModule <|-- BikeForecaster
    NnModule <|-- LSTMForecaster
    BikeForecasterTrainer --> BikeForecaster : builds
    BikeForecasterTrainer --> LSTMForecaster : builds
    BikeForecasterTrainer --> BikeDataset : uses (DataLoader)
    BikeDataProcessor ..> BikeForecasterTrainer : provides (X, y)
```

## 2. 클래스 다이어그램 — API / 서비스 레이어

```mermaid
classDiagram
    class FastAPIApp {
        -Dict STATE
        -Lock STATE_LOCK
        +reload_inference_model()
        +lifespan(app)
        +health() HealthResponse
        +current_metrics() ModelMetricsResponse
        +predict(req) PredictResponse
    }

    class DatasetsRouter {
        +get_summary() DatasetSummary
        +list_datasets() List
        +upload(file) UploadResponse
    }

    class TrainingRouter {
        +create_job(req) TrainingJobDTO
        +list_jobs() List
        +get_job(id) TrainingJobDTO
        +cancel(id)
        +stream_events(id) EventSourceResponse
    }

    class TrainingService {
        -Dict _REGISTRY
        -Lock _REGISTRY_LOCK
        +register_job(...) TrainingJob
        +get_job(id) TrainingJob
        +list_jobs() List
        +start_job(job, output_dir, on_completion)
        +cancel_job(id) bool
        +drain_events(job) AsyncGenerator
        -_run_training(job, output_dir, on_completion)
    }

    class TrainingJob {
        +str id
        +str model_type
        +int epochs
        +int batch_size
        +float learning_rate
        +str dataset_path
        +JobStatus status
        +float created_at
        +float started_at
        +float finished_at
        +Dict metrics
        +str error
        +List history
        -Queue _queue
        -EventLoop _loop
        -Event _cancel
        +to_dict() Dict
    }

    class JobStatus {
        <<enumeration>>
        PENDING
        RUNNING
        COMPLETED
        CANCELLED
        FAILED
    }

    class PydanticSchemas {
        <<pydantic models>>
        HealthResponse
        ModelMetricsResponse
        PredictRequest
        PredictResponse
        DatasetSummary
        UploadResponse
        TrainingCreateRequest
        TrainingJobDTO
    }

    FastAPIApp *-- DatasetsRouter : includes
    FastAPIApp *-- TrainingRouter : includes
    TrainingRouter --> TrainingService : uses
    DatasetsRouter ..> PydanticSchemas
    TrainingRouter ..> PydanticSchemas
    FastAPIApp ..> PydanticSchemas
    TrainingService "1" *-- "*" TrainingJob : owns
    TrainingJob --> JobStatus
    TrainingService ..> BikeForecasterTrainer : instantiates
    TrainingService ..> BikeDataProcessor : instantiates

    class BikeForecasterTrainer
    class BikeDataProcessor
```

## 3. 컴포넌트 다이어그램 — 시스템 전체

```mermaid
flowchart TB
    subgraph Browser["브라우저"]
        direction TB
        UI["React SPA<br/>(Vite + TS + Tailwind)"]
        subgraph Pages["페이지"]
            Home[HomePage]
            Train[TrainingPage]
            Pred[PredictPage]
        end
        RQ["TanStack Query<br/>(캐시/폴링)"]
        SSE["useTrainingStream<br/>(EventSource)"]
        APIC["api/client.ts<br/>(fetch 래퍼)"]
        UI --- Pages
        Pages --> RQ
        Pages --> SSE
        RQ --> APIC
    end

    subgraph Server["FastAPI Server (Uvicorn)"]
        direction TB
        Main["app/main.py<br/>CORS + STATE + Lock"]
        subgraph Routers["라우터"]
            RDS["routers/datasets.py"]
            RTR["routers/training.py"]
        end
        Svc["services/training.py<br/>TrainingJob 레지스트리"]
        subgraph ML["Core ML"]
            Proc[BikeDataProcessor]
            Trainer[BikeForecasterTrainer]
            Models["BikeForecaster /<br/>LSTMForecaster"]
        end
        Utils["utils.py<br/>save/load, metrics"]
        Main --- Routers
        RTR --> Svc
        Svc --> ML
        ML --> Utils
        Main --> Utils
    end

    subgraph FS["파일 시스템"]
        D1["data/train.csv"]
        D2["outputs/uploads/*.csv"]
        D3["outputs/models/*.pth<br/>+ preprocessor.joblib"]
        D4["outputs/predictions/*.json"]
    end

    APIC -- "REST /api/*" --> Main
    SSE -- "GET events (SSE)" --> RTR
    RDS --> D2
    Svc --> D1
    Svc --> D2
    Utils --> D3
    Utils --> D4
    Main --> D3
```

## 4. 시퀀스 다이어그램 — 학습 워크플로우

```mermaid
sequenceDiagram
    actor User as 사용자
    participant FE as Frontend (React)
    participant API as FastAPI
    participant Svc as TrainingService
    participant Th as Training Thread
    participant FS as File System

    User->>FE: CSV 선택 + 업로드
    FE->>API: POST /api/datasets/upload
    API->>FS: outputs/uploads/{id}.csv 저장
    API-->>FE: UploadResponse {dataset_id, stats}

    User->>FE: 모델/epochs/lr 설정 후 Start
    FE->>API: POST /api/training/jobs
    API->>Svc: register_job(...)
    Svc-->>API: TrainingJob (PENDING)
    API->>Svc: start_job(job, on_completion)
    Svc->>Th: spawn daemon thread (_run_training)
    API-->>FE: 202 Accepted + job_id

    FE->>API: GET /api/training/jobs/{id}/events (SSE open)
    API->>Svc: drain_events(job) async gen

    Th->>FS: CSV 읽기
    Th->>Th: BikeDataProcessor.prepare_data
    Th->>Th: trainer.build_model + train loop
    loop 매 epoch
        Th->>Th: forward / backward / validate
        Th->>Svc: on_epoch_end → queue에 push
        Svc-->>API: yield 'epoch' event
        API-->>FE: SSE 'epoch' {train_loss, val_loss}
        FE->>FE: 차트 업데이트
    end

    Th->>Th: predict(X_test) + calculate_metrics
    Th->>FS: save_model (.pth + .joblib)
    Th->>API: on_completion() → reload_inference_model()
    Th->>Svc: status=COMPLETED, emit 'done'
    Svc-->>API: yield 'done' event
    API-->>FE: SSE 'done' {metrics}
    FE->>FE: 최종 지표 표시, EventSource close
```

## 5. 시퀀스 다이어그램 — 예측 워크플로우

```mermaid
sequenceDiagram
    actor User as 사용자
    participant FE as Frontend (PredictPage)
    participant API as FastAPI
    participant State as STATE 캐시
    participant Model as PyTorch Model

    User->>FE: 폼 입력 (datetime, 기온, 날씨 등)
    FE->>API: POST /api/predict (PredictRequest)
    API->>State: acquire STATE_LOCK
    State-->>API: (model, processor, feature_cols)
    API->>API: DataFrame 생성
    API->>API: engineer_features()
    API->>API: scaler.transform(numerical)
    API->>API: feature 벡터 → tensor
    alt LSTM
        API->>API: tensor.unsqueeze(1)  # (1,1,24)
    else MLP
        API->>API: tensor shape (1,24)
    end
    API->>Model: forward (no_grad)
    Model-->>API: 예측값
    API->>API: max(0.0, y)  # 음수 클램핑
    API-->>FE: PredictResponse {count, count_rounded}
    FE-->>User: 결과 표시
```

## 6. 시퀀스 다이어그램 — 학습 취소

```mermaid
sequenceDiagram
    actor User as 사용자
    participant FE as Frontend
    participant API as FastAPI
    participant Svc as TrainingService
    participant Th as Training Thread

    User->>FE: Cancel 클릭
    FE->>API: DELETE /api/training/jobs/{id}
    API->>Svc: cancel_job(id)
    Svc->>Th: job._cancel.set()
    Svc-->>API: 200 OK
    API-->>FE: 성공

    Note over Th: 다음 epoch 경계에서 should_stop() True
    Th->>Svc: status=CANCELLED, emit 'cancelled'
    Svc-->>API: yield 'cancelled' event
    API-->>FE: SSE 'cancelled' {completed_epochs}
    FE->>FE: 상태 배지 갱신, EventSource close
```

## 7. 상태 다이어그램 — TrainingJob 라이프사이클

```mermaid
stateDiagram-v2
    [*] --> PENDING : register_job()
    PENDING --> RUNNING : start_job() → thread 시작
    RUNNING --> COMPLETED : 학습 루프 정상 종료 + 저장 완료
    RUNNING --> CANCELLED : cancel_job() → _cancel.set()
    RUNNING --> FAILED : 예외 발생
    COMPLETED --> [*]
    CANCELLED --> [*]
    FAILED --> [*]

    note right of RUNNING
        - epoch 이벤트를 큐에 enqueue
        - should_stop() 폴링
        - on_completion 콜백으로
          inference 모델 리로드
    end note
```

## 8. ER 스타일 — 데이터 스키마 (CSV 입력)

```mermaid
erDiagram
    BIKE_RECORD {
        datetime datetime PK "시각 (ISO)"
        int season "1=봄, 2=여름, 3=가을, 4=겨울"
        int holiday "0 또는 1"
        int workingday "0 또는 1"
        int weather "1=맑음, 2=흐림/안개, 3=약한 비/눈, 4=강한 비/눈"
        float temp "섭씨 기온"
        float atemp "체감 기온"
        float humidity "0-100"
        float windspeed "풍속"
        int count "타깃: 시간별 대여 수"
    }

    ENGINEERED_FEATURES {
        int hour
        int day
        int month
        int year
        int weekday
        float hour_sin
        float hour_cos
        float day_sin
        float day_cos
        float month_sin
        float month_cos
        float weekday_sin
        float weekday_cos
        int is_rush_hour
        int is_weekend
        float temp_humidity
        float temp_windspeed
    }

    BIKE_RECORD ||--|| ENGINEERED_FEATURES : "engineer_features()"
```

## 9. 활동 다이어그램 — `_run_training` 내부 플로우

```mermaid
flowchart TD
    Start([Thread 시작]) --> SetRun[status = RUNNING]
    SetRun --> LoadCSV[CSV 로드<br/>dataset_path 또는 기본값]
    LoadCSV --> Prep[BikeDataProcessor<br/>engineer_features + scale]
    Prep --> Split[time_series_split<br/>train / val / test]
    Split --> Build[trainer.build_model]
    Build --> Loop{epoch < max_epochs<br/>AND not should_stop?}
    Loop -- Yes --> FB[forward / backward / validate]
    FB --> Emit[on_epoch_end →<br/>queue enqueue]
    Emit --> Early{early stopping<br/>triggered?}
    Early -- No --> Loop
    Early -- Yes --> Predict[trainer.predict X_test]
    Loop -- No (cancel) --> Cancelled[status = CANCELLED<br/>emit 'cancelled']
    Loop -- No (done) --> Predict
    Predict --> Metrics[calculate_metrics]
    Metrics --> Save[save_model: .pth + .joblib]
    Save --> Reload[on_completion →<br/>reload_inference_model]
    Reload --> Done[status = COMPLETED<br/>emit 'done']
    Done --> End([Thread 종료])
    Cancelled --> End

    FB -. 예외 .-> Fail[status = FAILED<br/>emit 'error']
    Save -. 예외 .-> Fail
    Fail --> End
```

## 10. 패키지 다이어그램

```mermaid
flowchart LR
    subgraph backend_pkg["backend 패키지"]
        subgraph app_pkg["app"]
            appmain[main.py]
            subgraph routers_pkg["routers"]
                rds[datasets.py]
                rtr[training.py]
            end
            subgraph services_pkg["services"]
                stsvc[training.py]
            end
        end
        core[bike_forecast_pytorch.py]
        utils[utils.py]
        cfg[config.yaml]
    end

    subgraph frontend_pkg["frontend 패키지"]
        subgraph src_pkg["src"]
            maintsx[main.tsx]
            subgraph api_pkg["api"]
                client[client.ts]
                schema[schema.ts]
            end
            subgraph components_pkg["components"]
                appshell[AppShell.tsx]
                card[Card.tsx]
            end
            subgraph pages_pkg["pages"]
                home[HomePage.tsx]
                pred[PredictPage.tsx]
                train[TrainingPage.tsx]
            end
            subgraph hooks_pkg["hooks"]
                stream[useTrainingStream.ts]
            end
        end
    end

    appmain --> routers_pkg
    appmain --> core
    appmain --> utils
    stsvc --> core
    stsvc --> utils
    rtr --> stsvc
    rds --> utils
    core --> utils
    appmain -.읽기.-> cfg

    maintsx --> pages_pkg
    maintsx --> components_pkg
    pages_pkg --> api_pkg
    pages_pkg --> hooks_pkg
    train --> stream
    stream -.SSE.-> rtr
    client -.REST.-> appmain
```

## 11. 렌더링 방법

- **GitHub/GitLab**: `.md` 파일을 직접 열면 Mermaid 블록이 자동 렌더링됩니다.
- **VSCode**: "Markdown Preview Mermaid Support" 확장 설치 후 프리뷰(Ctrl+Shift+V).
- **IntelliJ / PyCharm**: Markdown 툴 창에서 Mermaid 지원 플러그인 활성화.
- **정적 사이트**: MkDocs-Material, Docusaurus 등 대부분의 도구가 Mermaid를 기본 지원합니다.
