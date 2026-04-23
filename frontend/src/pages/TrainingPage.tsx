import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useRef, useState } from 'react'
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, type TrainingCreateRequest, type UploadResponse } from '../api/client'
import { Card, Metric } from '../components/Card'
import { useTrainingStream, type StreamStatus } from '../hooks/useTrainingStream'

const BADGE_STYLES: Record<StreamStatus, string> = {
  idle: 'bg-slate-100 text-slate-600',
  running: 'bg-blue-100 text-blue-700',
  completed: 'bg-emerald-100 text-emerald-700',
  cancelled: 'bg-amber-100 text-amber-700',
  failed: 'bg-red-100 text-red-700',
}

function StatusBadge({ status }: { status: StreamStatus }) {
  if (status === 'idle') return null
  return (
    <span className={`mb-3 inline-block rounded px-2 py-0.5 text-xs font-medium ${BADGE_STYLES[status]}`}>
      {status}
    </span>
  )
}

const fieldCls =
  'mt-1 block w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm focus:border-slate-500 focus:outline-none'
const labelCls = 'text-xs font-medium uppercase tracking-wide text-slate-600'

export function TrainingPage() {
  const qc = useQueryClient()

  const [form, setForm] = useState<TrainingCreateRequest>({
    model_type: 'mlp',
    epochs: 30,
    batch_size: 128,
    learning_rate: 0.001,
    dataset_id: null,
  })
  const [jobId, setJobId] = useState<string | null>(null)
  const [upload, setUpload] = useState<UploadResponse | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const datasets = useQuery({ queryKey: ['datasets'], queryFn: api.listDatasets })

  const uploadMut = useMutation({
    mutationFn: api.uploadDataset,
    onSuccess: res => {
      setUpload(res)
      setForm(f => ({ ...f, dataset_id: res.dataset_id }))
      qc.invalidateQueries({ queryKey: ['datasets'] })
    },
  })

  const startMut = useMutation({
    mutationFn: api.startJob,
    onSuccess: job => {
      setJobId(job.id)
      qc.invalidateQueries({ queryKey: ['metrics'] })
    },
  })

  const cancelMut = useMutation({
    mutationFn: api.cancelJob,
  })

  const stream = useTrainingStream(jobId)

  const chartData = useMemo(
    () => stream.epochs.map(e => ({ epoch: e.epoch, train: e.train_loss, val: e.val_loss })),
    [stream.epochs],
  )

  const onFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) uploadMut.mutate(f)
  }

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    startMut.mutate(form)
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_2fr]">
      <Card title="Training config">
        <form onSubmit={onSubmit} className="grid gap-4">
          <label>
            <span className={labelCls}>Dataset</span>
            <select
              className={fieldCls}
              value={form.dataset_id ?? ''}
              onChange={e => setForm({ ...form, dataset_id: e.target.value || null })}
            >
              <option value="">Default (train.csv)</option>
              {datasets.data?.map(d => (
                <option key={d.dataset_id} value={d.dataset_id}>
                  {d.filename} ({(d.size_bytes / 1024).toFixed(0)} KB)
                </option>
              ))}
            </select>
            <div className="mt-2 flex items-center gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={onFile}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploadMut.isPending}
                className="text-xs text-slate-600 underline underline-offset-2 hover:text-slate-900"
              >
                {uploadMut.isPending ? 'Uploading…' : '+ Upload CSV'}
              </button>
              {upload && (
                <span className="text-xs text-slate-500">
                  Uploaded {upload.rows.toLocaleString()} rows
                </span>
              )}
              {uploadMut.isError && (
                <span className="text-xs text-red-600">
                  {(uploadMut.error as Error).message}
                </span>
              )}
            </div>
          </label>

          <label>
            <span className={labelCls}>Model</span>
            <select
              className={fieldCls}
              value={form.model_type}
              onChange={e => setForm({ ...form, model_type: e.target.value as 'mlp' | 'lstm' })}
            >
              <option value="mlp">MLP</option>
              <option value="lstm">LSTM</option>
            </select>
          </label>

          <label>
            <span className={labelCls}>Epochs</span>
            <input
              type="number" min={1} max={200}
              className={fieldCls}
              value={form.epochs}
              onChange={e => setForm({ ...form, epochs: Number(e.target.value) })}
            />
          </label>

          <label>
            <span className={labelCls}>Batch size</span>
            <input
              type="number" min={8} max={512}
              className={fieldCls}
              value={form.batch_size}
              onChange={e => setForm({ ...form, batch_size: Number(e.target.value) })}
            />
          </label>

          <label>
            <span className={labelCls}>Learning rate</span>
            <input
              type="number" step="0.0001" min={0.0001} max={0.1}
              className={fieldCls}
              value={form.learning_rate}
              onChange={e => setForm({ ...form, learning_rate: Number(e.target.value) })}
            />
          </label>

          <div className="flex gap-2">
            <button
              type="submit"
              disabled={startMut.isPending || stream.status === 'running'}
              className="flex-1 rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
            >
              {stream.status === 'running' ? 'Training…' : 'Start training'}
            </button>
            {stream.status === 'running' && jobId && (
              <button
                type="button"
                onClick={() => cancelMut.mutate(jobId)}
                disabled={cancelMut.isPending}
                className="rounded border border-red-300 bg-white px-4 py-2 text-sm font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
              >
                {cancelMut.isPending ? 'Cancelling…' : 'Cancel'}
              </button>
            )}
          </div>

          {startMut.isError && (
            <div className="text-xs text-red-600">{(startMut.error as Error).message}</div>
          )}
          {cancelMut.isError && (
            <div className="text-xs text-red-600">Cancel failed: {(cancelMut.error as Error).message}</div>
          )}
        </form>
      </Card>

      <Card title={`Progress ${jobId ? `· ${jobId}` : ''}`}>
        <StatusBadge status={stream.status} />

        {stream.status === 'idle' && (
          <div className="text-slate-400">Start a training run to see live progress.</div>
        )}

        {stream.status === 'failed' && (
          <div className="text-red-600 text-sm">Error: {stream.error}</div>
        )}

        {stream.status === 'cancelled' && stream.cancelled && (
          <div className="mt-2 text-sm text-amber-700">
            Cancelled after {stream.cancelled.completed_epochs} epoch(s). The previous model was kept intact.
          </div>
        )}

        {chartData.length > 0 && (
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(2) : String(v))} />
                <Legend />
                <Line
                  type="monotone" dataKey="train" name="train loss"
                  stroke="#2563eb" strokeWidth={2} dot={false} isAnimationActive={false}
                />
                <Line
                  type="monotone" dataKey="val" name="val loss"
                  stroke="#dc2626" strokeWidth={2} dot={false} isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {stream.result && (
          <div className="mt-4 grid grid-cols-4 gap-6 border-t pt-4">
            <Metric label="R²" value={stream.result.metrics.r2.toFixed(3)} />
            <Metric label="RMSE" value={stream.result.metrics.rmse.toFixed(1)} />
            <Metric label="MAE" value={stream.result.metrics.mae.toFixed(1)} />
            <Metric
              label="Epochs"
              value={stream.epochs.length}
              hint={`best val: ${stream.result.metrics.rmse.toFixed(1)}`}
            />
          </div>
        )}
      </Card>
    </div>
  )
}
