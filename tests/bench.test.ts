import { test } from 'vitest';
import { Runner } from '../src/runner';
import { SCENARIOS } from '../src/world';
test.skipIf(!process.env.BENCH)('bench', () => {
  for (const id of ['open-sea', 'cyclone']) {
    const r = new Runner(SCENARIOS.find((s) => s.id === id)!);
    const core = r.model.core as any, film = (r.model as any).film;
    const T: Record<string, number> = {};
    const wrap = (o: any, k: string, name: string) => { const f = o[k].bind(o); o[k] = (...a: any[]) => { const t = performance.now(); const v = f(...a); T[name] = (T[name] ?? 0) + performance.now() - t; return v; }; };
    const time = (n: number) => { for (const k in T) delete T[k]; const t = performance.now(); for (let i = 0; i < n; i++) r.step(); return (performance.now() - t) / n; };
    for (let i = 0; i < 3; i++) r.step();
    wrap(core.water, 'step', 'water'); wrap(core, 'targets', 'targets'); wrap(core.oil, 'step', 'oil'); wrap(film, 'apply', 'film');
    const early = time(20); const e = { ...T };
    for (let i = 0; i < 130; i++) r.step();
    const late = time(10);
    const f = (o: Record<string, number>, n: number) => Object.entries(o).map(([k, v]) => `${k} ${(v / n).toFixed(1)}`).join(' ');
    console.log(`${id} early ${early.toFixed(1)} ms (${f(e, 20)}) | late ${late.toFixed(1)} ms (${f(T, 10)})`);
  }
}, 600000);
