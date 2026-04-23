"""Background training job runner with SSE-friendly event queues.

Each job runs in its own daemon thread (torch releases the GIL for kernels
so FastAPI stays responsive). Epoch events are pushed back to the event loop
via ``loop.call_soon_threadsafe`` into an ``asyncio.Queue`` that the SSE
endpoint drains.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from bike_forecast_pytorch import BikeDataProcessor, BikeForecasterTrainer
from utils import save_model, time_series_split, calculate_metrics


class JobStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    CANCELLED = 'cancelled'
    FAILED = 'failed'


@dataclass
class TrainingJob:
    id: str
    model_type: str
    epochs: int
    batch_size: int
    learning_rate: float
    dataset_path: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    metrics: dict | None = None
    error: str | None = None
    history: list[dict] = field(default_factory=list)
    _queue: asyncio.Queue = field(repr=False, default=None)
    _loop: asyncio.AbstractEventLoop = field(repr=False, default=None)
    _cancel: threading.Event = field(repr=False, default_factory=threading.Event)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'model_type': self.model_type,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'metrics': self.metrics,
            'error': self.error,
            'history': self.history,
        }


_REGISTRY: dict[str, TrainingJob] = {}
_REGISTRY_LOCK = threading.Lock()


def register_job(
    model_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dataset_path: str,
) -> TrainingJob:
    job = TrainingJob(
        id=uuid4().hex[:12],
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dataset_path=dataset_path,
    )
    with _REGISTRY_LOCK:
        _REGISTRY[job.id] = job
    return job


def get_job(job_id: str) -> TrainingJob | None:
    return _REGISTRY.get(job_id)


def list_jobs() -> list[TrainingJob]:
    with _REGISTRY_LOCK:
        return sorted(_REGISTRY.values(), key=lambda j: j.created_at, reverse=True)


def start_job(job: TrainingJob, output_dir: Path, on_completion: Callable[[], None] | None = None) -> None:
    """Launch the training thread. Must be called from inside an asyncio loop."""
    job._loop = asyncio.get_running_loop()
    job._queue = asyncio.Queue()

    thread = threading.Thread(
        target=_run_training,
        args=(job, output_dir, on_completion),
        daemon=True,
        name=f'train-{job.id}',
    )
    thread.start()


def _emit(job: TrainingJob, event: dict) -> None:
    """Push an SSE event from the training thread into the asyncio queue."""
    if job._loop is None or job._queue is None:
        return
    try:
        job._loop.call_soon_threadsafe(job._queue.put_nowait, event)
    except RuntimeError:
        # Loop closed — server shutting down; drop the event.
        pass


async def drain_events(job: TrainingJob):
    """Yield events for an SSE subscriber. Ends on done/cancelled/error."""
    terminal = {'done', 'cancelled', 'error'}
    if job._queue is None:
        return
    # First replay history in case the subscriber connected after training started.
    for epoch_event in job.history:
        yield {'event': 'epoch', 'data': epoch_event}
    if job.status is JobStatus.COMPLETED:
        yield {'event': 'done', 'data': {'metrics': job.metrics}}
        return
    if job.status is JobStatus.CANCELLED:
        yield {'event': 'cancelled', 'data': {'completed_epochs': len(job.history)}}
        return
    if job.status is JobStatus.FAILED:
        yield {'event': 'error', 'data': {'error': job.error}}
        return

    while True:
        event = await job._queue.get()
        if event is None:
            return
        yield event
        if event['event'] in terminal:
            return


def cancel_job(job_id: str) -> TrainingJob | None:
    """Signal the training thread to stop. Idempotent."""
    job = _REGISTRY.get(job_id)
    if job is None:
        return None
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        return job
    job._cancel.set()
    return job


# ---------- Training implementation ----------

def _load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def _run_training(job: TrainingJob, output_dir: Path, on_completion: Callable[[], None] | None) -> None:
    try:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        _emit(job, {'event': 'status', 'data': {'status': 'running'}})

        df = _load_dataset(job.dataset_path)
        processor = BikeDataProcessor()
        X, y, feature_cols = processor.prepare_data(df, target_col='count')
        X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(
            X, y, test_size=0.15, val_size=0.15
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = BikeForecasterTrainer(model_type=job.model_type, device=device)
        trainer.build_model(input_size=X.shape[1])

        def on_epoch(ev: dict) -> None:
            job.history.append(ev)
            _emit(job, {'event': 'epoch', 'data': ev})

        # num_workers=0 when spawned inside a daemon thread to avoid forking issues
        train_result = trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=job.epochs,
            batch_size=job.batch_size,
            lr=job.learning_rate,
            num_workers=0,
            on_epoch_end=on_epoch,
            should_stop=job._cancel.is_set,
        )

        if train_result and train_result.get('cancelled'):
            # User cancelled: keep the previous checkpoint untouched.
            job.status = JobStatus.CANCELLED
            job.finished_at = time.time()
            _emit(job, {
                'event': 'cancelled',
                'data': {'completed_epochs': train_result.get('completed_epochs', 0)},
            })
            return

        y_pred = trainer.predict(X_test)
        metrics = calculate_metrics(np.asarray(y_test), np.asarray(y_pred))
        job.metrics = metrics

        # Persist checkpoint so the predict API can pick it up on restart.
        model_dir = output_dir / 'models'
        save_model(
            trainer.model,
            str(model_dir / f'{job.model_type}_model.pth'),
            metadata={
                'model_type': job.model_type,
                'metrics': metrics,
                'feature_names': feature_cols,
                'training_time': time.time() - (job.started_at or time.time()),
                'job_id': job.id,
            },
            processor=processor,
            feature_cols=feature_cols,
        )

        job.status = JobStatus.COMPLETED
        job.finished_at = time.time()
        _emit(job, {'event': 'done', 'data': {'metrics': metrics}})
        if on_completion is not None:
            on_completion()
    except Exception as exc:  # noqa: BLE001
        job.status = JobStatus.FAILED
        job.finished_at = time.time()
        job.error = f'{exc.__class__.__name__}: {exc}'
        traceback.print_exc()
        _emit(job, {'event': 'error', 'data': {'error': job.error}})
