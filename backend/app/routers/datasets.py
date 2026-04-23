"""Dataset upload + summary routes."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

if TYPE_CHECKING:
    pass

router = APIRouter(prefix='/api/datasets', tags=['datasets'])

REQUIRED_COLUMNS = {
    'datetime', 'season', 'holiday', 'workingday', 'weather',
    'temp', 'atemp', 'humidity', 'windspeed', 'count',
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class DatasetSummary(BaseModel):
    rows: int
    columns: int
    date_range: dict[str, str]
    target: dict[str, float]


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    date_range: dict[str, str]
    target: dict[str, float]


class DatasetListItem(BaseModel):
    dataset_id: str
    filename: str
    size_bytes: int


def _train_csv_path() -> Path:
    from app.main import TRAIN_CSV
    return TRAIN_CSV


def _upload_dir() -> Path:
    from app.main import UPLOAD_DIR
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def _summarise(df: pd.DataFrame) -> dict:
    return {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'date_range': {
            'start': df['datetime'].min().isoformat(),
            'end': df['datetime'].max().isoformat(),
        },
        'target': {
            'mean': float(df['count'].mean()),
            'std': float(df['count'].std()),
            'min': float(df['count'].min()),
            'max': float(df['count'].max()),
        },
    }


@router.get('/summary', response_model=DatasetSummary)
async def summary() -> DatasetSummary:
    path = _train_csv_path()
    if not path.exists():
        raise HTTPException(404, 'train.csv not found')
    df = pd.read_csv(path, parse_dates=['datetime'])
    return DatasetSummary(**_summarise(df))


@router.get('', response_model=list[DatasetListItem])
async def list_uploads() -> list[DatasetListItem]:
    items: list[DatasetListItem] = []
    for p in sorted(_upload_dir().glob('*.csv')):
        items.append(DatasetListItem(
            dataset_id=p.stem,
            filename=p.name,
            size_bytes=p.stat().st_size,
        ))
    return items


@router.post('/upload', response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not (file.filename or '').lower().endswith('.csv'):
        raise HTTPException(400, 'Only .csv files are accepted')

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f'File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit')

    tmp_path = _upload_dir() / f'_tmp_{uuid.uuid4().hex}.csv'
    tmp_path.write_bytes(contents)
    try:
        df = pd.read_csv(tmp_path, parse_dates=['datetime'])
    except Exception as exc:  # noqa: BLE001
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(400, f'Failed to parse CSV: {exc}') from exc

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(400, f'Missing required columns: {sorted(missing)}')

    dataset_id = uuid.uuid4().hex[:12]
    final_path = _upload_dir() / f'{dataset_id}.csv'
    tmp_path.rename(final_path)

    summary = _summarise(df)
    return UploadResponse(
        dataset_id=dataset_id,
        filename=file.filename or final_path.name,
        rows=summary['rows'],
        date_range=summary['date_range'],
        target=summary['target'],
    )
