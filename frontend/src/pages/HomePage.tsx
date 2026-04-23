import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { Card, Metric } from '../components/Card'

export function HomePage() {
  const health = useQuery({ queryKey: ['health'], queryFn: api.health })
  const metrics = useQuery({ queryKey: ['metrics'], queryFn: api.metrics })
  const summary = useQuery({ queryKey: ['summary'], queryFn: api.datasetSummary })

  return (
    <div className="grid gap-6">
      <Card title="Model status">
        {health.isLoading ? (
          <div className="text-slate-400">Loading...</div>
        ) : health.isError ? (
          <div className="text-red-600">API unreachable: {(health.error as Error).message}</div>
        ) : (
          <div className="grid grid-cols-3 gap-6">
            <Metric label="Status" value={health.data!.status} />
            <Metric label="Model" value={health.data!.model_class ?? '—'} />
            <Metric label="Input features" value={health.data!.input_size ?? '—'} />
          </div>
        )}
      </Card>

      <Card title="Test-set performance">
        {metrics.isLoading ? (
          <div className="text-slate-400">Loading...</div>
        ) : metrics.isError ? (
          <div className="text-red-600">{(metrics.error as Error).message}</div>
        ) : (
          <div className="grid grid-cols-4 gap-6">
            <Metric
              label="R²"
              value={metrics.data!.metrics.r2.toFixed(3)}
              hint="higher is better"
            />
            <Metric label="RMSE" value={metrics.data!.metrics.rmse.toFixed(1)} hint="count units" />
            <Metric label="MAE" value={metrics.data!.metrics.mae.toFixed(1)} hint="count units" />
            <Metric label="n samples" value={metrics.data!.n_samples.toLocaleString()} />
          </div>
        )}
      </Card>

      <Card title="Training dataset">
        {summary.isLoading ? (
          <div className="text-slate-400">Loading...</div>
        ) : summary.isError ? (
          <div className="text-red-600">{(summary.error as Error).message}</div>
        ) : (
          <div className="grid grid-cols-4 gap-6">
            <Metric label="Rows" value={summary.data!.rows.toLocaleString()} />
            <Metric
              label="Date range"
              value={
                <span className="text-sm">
                  {summary.data!.date_range.start.slice(0, 10)} →<br />
                  {summary.data!.date_range.end.slice(0, 10)}
                </span>
              }
            />
            <Metric label="Target mean" value={summary.data!.target.mean.toFixed(1)} />
            <Metric label="Target max" value={summary.data!.target.max.toFixed(0)} />
          </div>
        )}
      </Card>
    </div>
  )
}
