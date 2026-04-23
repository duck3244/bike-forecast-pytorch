import type { components } from './schema'

export type HealthResponse = components['schemas']['HealthResponse']
export type ModelMetricsResponse = components['schemas']['ModelMetricsResponse']
export type DatasetSummary = components['schemas']['DatasetSummary']
export type UploadResponse = components['schemas']['UploadResponse']
export type DatasetListItem = components['schemas']['DatasetListItem']
export type PredictRequest = components['schemas']['PredictRequest']
export type PredictResponse = components['schemas']['PredictResponse']
export type TrainingCreateRequest = components['schemas']['TrainingCreateRequest']
export type TrainingJobDTO = components['schemas']['TrainingJobDTO']

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...(init?.headers ?? {}),
    },
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<HealthResponse>('/api/health'),
  metrics: () => request<ModelMetricsResponse>('/api/models/current/metrics'),
  datasetSummary: () => request<DatasetSummary>('/api/datasets/summary'),
  predict: (body: PredictRequest) =>
    request<PredictResponse>('/api/predict', { method: 'POST', body: JSON.stringify(body) }),

  listDatasets: () => request<DatasetListItem[]>('/api/datasets'),
  uploadDataset: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return request<UploadResponse>('/api/datasets/upload', { method: 'POST', body: fd })
  },

  listJobs: () => request<TrainingJobDTO[]>('/api/training/jobs'),
  getJob: (id: string) => request<TrainingJobDTO>(`/api/training/jobs/${id}`),
  startJob: (body: TrainingCreateRequest) =>
    request<TrainingJobDTO>('/api/training/jobs', { method: 'POST', body: JSON.stringify(body) }),
  cancelJob: (id: string) =>
    request<TrainingJobDTO>(`/api/training/jobs/${id}`, { method: 'DELETE' }),
}
