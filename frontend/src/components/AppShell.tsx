import { NavLink, Outlet } from 'react-router-dom'

const navLinkCls = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-2 text-sm font-medium rounded transition ${
    isActive ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-200'
  }`

export function AppShell() {
  return (
    <div className="min-h-screen w-full bg-slate-50 text-slate-900">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <span className="text-xl">🚴</span>
            <h1 className="text-lg font-semibold">Bike Demand Forecast</h1>
          </div>
          <nav className="flex gap-2">
            <NavLink to="/" end className={navLinkCls}>Dashboard</NavLink>
            <NavLink to="/train" className={navLinkCls}>Train</NavLink>
            <NavLink to="/predict" className={navLinkCls}>Predict</NavLink>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  )
}
