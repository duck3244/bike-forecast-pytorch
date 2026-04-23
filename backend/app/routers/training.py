"""Training job routes: start, status, SSE progress stream."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.services import training as training_service

router = APIRouter(prefix='/api/training', tags=['training'])


class TrainingCreateRequest(BaseModel):
    model_type: Literal['mlp', 'lstm'] = 'mlp'
    epochs: int = Field(20, ge=1, le=200)
    batch_size: int = Field(64, ge=8, le=512)
    learning_rate: float = Field(0.001, gt=0, le=0.1)
    dataset_id: str | None = None


class TrainingJobDTO(BaseModel):
    id: str
    model_type: str
    epochs: int
    batch_size: int
    learning_rate: float
    status: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    metrics: dict | None
    error: str | None
    history: list[dict]


def _resolve_dataset(dataset_id: str | None) -> str:
    from app.main import TRAIN_CSV, UPLOAD_DIR
    if dataset_id is None:
        if not TRAIN_CSV.exists():
            raise HTTPException(404, 'Default train.csv not found')
        return str(TRAIN_CSV)
    path = UPLOAD_DIR / f'{dataset_id}.csv'
    if not path.exists():
        raise HTTPException(404, f'Dataset {dataset_id} not found')
    return str(path)


@router.post('/jobs', response_model=TrainingJobDTO, status_code=202)
async def create_job(body: TrainingCreateRequest) -> TrainingJobDTO:
    from app.main import OUTPUT_DIR, reload_inference_model

    dataset_path = _resolve_dataset(body.dataset_id)
    job = training_service.register_job(
        model_type=body.model_type,
        epochs=body.epochs,
        batch_size=body.batch_size,
        learning_rate=body.learning_rate,
        dataset_path=dataset_path,
    )
    training_service.start_job(job, OUTPUT_DIR, on_completion=reload_inference_model)
    return TrainingJobDTO(**job.to_dict())


@router.get('/jobs', response_model=list[TrainingJobDTO])
async def list_jobs() -> list[TrainingJobDTO]:
    return [TrainingJobDTO(**j.to_dict()) for j in training_service.list_jobs()]


@router.get('/jobs/{job_id}', response_model=TrainingJobDTO)
async def get_job(job_id: str) -> TrainingJobDTO:
    job = training_service.get_job(job_id)
    if job is None:
        raise HTTPException(404, 'Job not found')
    return TrainingJobDTO(**job.to_dict())


@router.delete('/jobs/{job_id}', response_model=TrainingJobDTO)
async def cancel_job(job_id: str) -> TrainingJobDTO:
    job = training_service.cancel_job(job_id)
    if job is None:
        raise HTTPException(404, 'Job not found')
    return TrainingJobDTO(**job.to_dict())


@router.get('/jobs/{job_id}/events')
async def job_events(job_id: str, request: Request):
    job = training_service.get_job(job_id)
    if job is None:
        raise HTTPException(404, 'Job not found')

    async def event_stream():
        async for event in training_service.drain_events(job):
            if await request.is_disconnected():
                break
            yield {
                'event': event['event'],
                'data': __import__('json').dumps(event['data']),
            }
            # Give the event loop a chance to flush
            await asyncio.sleep(0)

    return EventSourceResponse(event_stream())
