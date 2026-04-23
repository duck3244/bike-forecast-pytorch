import { useEffect, useRef, useState } from 'react'

export interface EpochEvent {
  epoch: number
  train_loss: number
  val_loss: number
  best_val_loss: number
  is_best: boolean
}

export interface DoneEvent {
  metrics: {
    r2: number
    rmse: number
    mae: number
    mse: number
    mape: number | null
  }
}

export interface CancelledEvent {
  completed_epochs: number
}

export type StreamStatus = 'idle' | 'running' | 'completed' | 'cancelled' | 'failed'

export interface TrainingStreamState {
  status: StreamStatus
  epochs: EpochEvent[]
  result: DoneEvent | null
  cancelled: CancelledEvent | null
  error: string | null
}

const initial: TrainingStreamState = {
  status: 'idle',
  epochs: [],
  result: null,
  cancelled: null,
  error: null,
}

export function useTrainingStream(jobId: string | null) {
  const [state, setState] = useState<TrainingStreamState>(initial)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!jobId) {
      setState(initial)
      return
    }
    setState({ ...initial, status: 'running' })

    const es = new EventSource(`/api/training/jobs/${jobId}/events`)
    esRef.current = es

    es.addEventListener('epoch', e => {
      const data = JSON.parse((e as MessageEvent).data) as EpochEvent
      setState(s => ({ ...s, epochs: [...s.epochs, data] }))
    })
    es.addEventListener('done', e => {
      const data = JSON.parse((e as MessageEvent).data) as DoneEvent
      setState(s => ({ ...s, status: 'completed', result: data }))
      es.close()
    })
    es.addEventListener('cancelled', e => {
      const data = JSON.parse((e as MessageEvent).data) as CancelledEvent
      setState(s => ({ ...s, status: 'cancelled', cancelled: data }))
      es.close()
    })
    es.addEventListener('error', e => {
      const msg = e instanceof MessageEvent && e.data
        ? (JSON.parse(e.data as string)?.error ?? 'stream error')
        : 'connection lost'
      setState(s => ({ ...s, status: 'failed', error: msg }))
      es.close()
    })

    return () => {
      es.close()
    }
  }, [jobId])

  return state
}
