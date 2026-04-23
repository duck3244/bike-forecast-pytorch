import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'
import { api, type PredictRequest } from '../api/client'
import { Card, Metric } from '../components/Card'

const defaults: PredictRequest = {
  datetime: '2011-07-15T08:00:00',
  season: 3,
  holiday: 0,
  workingday: 1,
  weather: 1,
  temp: 28.5,
  atemp: 32.0,
  humidity: 60,
  windspeed: 12.5,
}

const SEASONS = { 1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter' }
const WEATHERS = { 1: 'Clear', 2: 'Mist', 3: 'Light rain/snow', 4: 'Heavy rain/snow' }

const fieldCls =
  'mt-1 block w-full rounded border border-slate-300 bg-white px-3 py-2 text-sm focus:border-slate-500 focus:outline-none'
const labelCls = 'text-xs font-medium uppercase tracking-wide text-slate-600'

export function PredictPage() {
  const [form, setForm] = useState<PredictRequest>(defaults)
  const mutation = useMutation({ mutationFn: api.predict })

  const onChange = <K extends keyof PredictRequest>(key: K, value: PredictRequest[K]) =>
    setForm(prev => ({ ...prev, [key]: value }))

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    mutation.mutate(form)
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
      <Card title="Input features">
        <form onSubmit={submit} className="grid grid-cols-2 gap-4">
          <label className="col-span-2">
            <span className={labelCls}>Datetime</span>
            <input
              type="datetime-local"
              className={fieldCls}
              value={form.datetime.slice(0, 16)}
              onChange={e => onChange('datetime', `${e.target.value}:00`)}
            />
          </label>

          <label>
            <span className={labelCls}>Season</span>
            <select
              className={fieldCls}
              value={form.season}
              onChange={e => onChange('season', Number(e.target.value))}
            >
              {Object.entries(SEASONS).map(([v, n]) => (
                <option key={v} value={v}>{v} · {n}</option>
              ))}
            </select>
          </label>

          <label>
            <span className={labelCls}>Weather</span>
            <select
              className={fieldCls}
              value={form.weather}
              onChange={e => onChange('weather', Number(e.target.value))}
            >
              {Object.entries(WEATHERS).map(([v, n]) => (
                <option key={v} value={v}>{v} · {n}</option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={form.holiday === 1}
              onChange={e => onChange('holiday', e.target.checked ? 1 : 0)}
            />
            <span className={labelCls}>Holiday</span>
          </label>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={form.workingday === 1}
              onChange={e => onChange('workingday', e.target.checked ? 1 : 0)}
            />
            <span className={labelCls}>Working day</span>
          </label>

          <label>
            <span className={labelCls}>Temp (°C)</span>
            <input
              type="number" step="0.1" className={fieldCls}
              value={form.temp}
              onChange={e => onChange('temp', Number(e.target.value))}
            />
          </label>

          <label>
            <span className={labelCls}>Feels like (°C)</span>
            <input
              type="number" step="0.1" className={fieldCls}
              value={form.atemp}
              onChange={e => onChange('atemp', Number(e.target.value))}
            />
          </label>

          <label>
            <span className={labelCls}>Humidity (%)</span>
            <input
              type="number" step="1" min="0" max="100" className={fieldCls}
              value={form.humidity}
              onChange={e => onChange('humidity', Number(e.target.value))}
            />
          </label>

          <label>
            <span className={labelCls}>Wind speed</span>
            <input
              type="number" step="0.1" min="0" className={fieldCls}
              value={form.windspeed}
              onChange={e => onChange('windspeed', Number(e.target.value))}
            />
          </label>

          <div className="col-span-2 mt-2">
            <button
              type="submit"
              disabled={mutation.isPending}
              className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
            >
              {mutation.isPending ? 'Predicting…' : 'Predict'}
            </button>
          </div>
        </form>
      </Card>

      <Card title="Prediction">
        {mutation.isIdle && <div className="text-slate-400">Fill the form and press Predict.</div>}
        {mutation.isError && (
          <div className="text-red-600 text-sm">{(mutation.error as Error).message}</div>
        )}
        {mutation.data && (
          <div className="grid gap-4">
            <Metric
              label="Predicted rentals"
              value={mutation.data.count_rounded.toLocaleString()}
              hint={`raw: ${mutation.data.count.toFixed(2)}`}
            />
          </div>
        )}
      </Card>
    </div>
  )
}
