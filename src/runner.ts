// Hosts the slick model on a scenario's coast and applies edits. Pure: no DOM, no timers, so the worker, the
// determinism check and the test all run the same code.
import { MasterModel } from './engine/film/master';
import { DEFAULT_COAST, OilLayer, ShelfWater } from './engine/film/twolayer';
import { Budget, volume } from './engine/film/grid';
import { SurfaceFlow } from './engine/film/flow';
import { subSeed } from './engine/rng/prng';
import { polygon, stroke } from './engine/film/shapes';
import type { Budget as BudgetMsg } from './protocol';
import { CHUNK, DX, HALF, N, landMask, type Edit, type Pt, type Scenario, type Scene, type Boom } from './world';

/** Changing these rebuilds the eddy field, which is drawn from them once. */
const FLOW_SHAPE = new Set(['eddySpeed', 'eddyScaleM']);
const DEPTH = 12;
/** Steps between refreshes of the drift the tracers follow: it is for show, and costs a full flow evaluation. */
const DRIFT_EVERY = 10;

export const circle = ([cx, cy]: Pt, r: number) => {
  const flat: number[] = [];
  for (let s = 0; s < 24; s++) { const a = (s / 24) * 2 * Math.PI; flat.push(cx + r * Math.cos(a), cy + r * Math.sin(a)); }
  return polygon(flat);
};

/** Cells a boom closes: sampled along each segment and a little to either side, so it is at least two cells wide. */
export function boomCells(booms: Boom[]): Uint8Array {
  const m = new Uint8Array(N * N);
  const mark = (x: number, y: number) => {
    const i = Math.floor((x + HALF) / DX), j = Math.floor((y + HALF) / DX);
    if (i >= 0 && j >= 0 && i < N && j < N) m[j * N + i] = 1;
  };
  for (const b of booms)
    for (let s = 1; s < b.path.length; s++) {
      const [ax, ay] = b.path[s - 1], [bx, by] = b.path[s], L = Math.hypot(bx - ax, by - ay);
      if (!L) continue;
      const nx = (-(by - ay) / L) * DX * 0.35, ny = ((bx - ax) / L) * DX * 0.35;
      for (let d = 0; d <= L; d += DX / 4) {
        const x = ax + ((bx - ax) * d) / L, y = ay + ((by - ay) * d) / L;
        mark(x, y); mark(x + nx, y + ny); mark(x - nx, y - ny);
      }
    }
  return m;
}

/** FNV-1a over the raw bits of the oil thickness and momentum: equal only if every cell is bit-identical. */
export function fingerprint(...fields: Float64Array[]): string {
  let a = 0x811c9dc5;
  for (const f of fields) {
    const u = new Uint32Array(f.buffer, f.byteOffset, f.length * 2);
    for (let i = 0; i < u.length; i++) a = Math.imul(a ^ u[i], 0x01000193);
  }
  return (a >>> 0).toString(16).padStart(8, '0');
}

export class Runner {
  readonly model: MasterModel;
  readonly base: Uint8Array;
  chunk = 0;
  scene: Scene = { booms: [], strokes: [], sources: [] };
  /** Oil caught where a boom was laid over it: kept in the budget as stranded. */
  heldByBooms = 0;

  constructor(readonly sc: Scenario) {
    this.model = new MasterModel(sc.params as Partial<MasterModel["params"]>);
    const core = this.model.core;
    this.base = landMask(sc);
    core.oil = new OilLayer(N, N, DX, this.base.slice());
    core.budget = new Budget(0);
    // The water layer is 128² at 100 m: a water cell is land when most of its four oil cells are. A 12 m shelf
    // everywhere: its stable step is set by the deepest water, so the engine's 30 m default coast would cost
    // half again as many water steps for the same flow.
    const w = (core.water = new ShelfWater(128, 100, { ...DEFAULT_COAST, hMax: DEPTH }));
    for (let j = 0; j < w.n; j++)
      for (let i = 0; i < w.n; i++) {
        const b = this.base, k = 2 * j * N + 2 * i;
        w.land[j * w.n + i] = b[k] + b[k + 1] + b[k + N] + b[k + N + 1] >= 2 ? 1 : 0;
        w.H[j * w.n + i] = DEPTH;
      }
    this.apply({ k: 'scene', scene: sc.scene });
    for (const spill of sc.spills) this.apply({ k: 'spill', spill });
    this.drift = this.computeDrift();
  }

  get t() { return this.model.t; }

  apply(e: Edit) {
    const m = this.model, core = m.core;
    if (e.k === 'param') {
      m.params[e.key] = e.value;
      const old = core.flow;
      if (FLOW_SHAPE.has(e.key)) {
        core.flow = new SurfaceFlow(m.params, subSeed(m.params.seed, 1));
        core.flow.setRegions(core.regions);
      }
      // The core caches the eddy and windrow field by version: a new wind must show now, not at the next refresh.
      core.flow.version = old.version + 1;
    } else if (e.k === 'spill') {
      m.addOil(circle(e.spill.at, e.spill.radius), e.spill.volume, 'dome', false);
    } else {
      this.scene = structuredClone(e.scene);
      m.setRegions(this.scene.strokes.map((s) => stroke(s.id, s.kind, s.path.flat(), s.width, s.speed)));
      const o = core.oil, booms = boomCells(this.scene.booms);
      for (let k = 0; k < o.land.length; k++) {
        const land = this.base[k] | booms[k];
        if (land === o.land[k]) continue;
        if (land && o.h[k] > 0) {
          const v = o.h[k] * DX * DX;
          o.stranded[k] += v; core.budget.stranded += v; this.heldByBooms += v;
        }
        o.h[k] = 0; o.qx[k] = 0; o.qy[k] = 0;
        o.land[k] = land;
      }
    }
  }

  step() {
    for (const s of this.scene.sources) this.model.addOil(circle(s.at, 90), (s.rate * CHUNK) / 3600, 'uniform', false);
    this.model.step(CHUNK);
    this.chunk++;
    if (this.chunk % DRIFT_EVERY === 0) this.drift = this.computeDrift();
  }

  warnings() { return this.model.warnings(); }

  fingerprint() { return fingerprint(this.model.core.oil.h, this.model.core.oil.qx, this.model.core.oil.qy); }

  floating() { return volume(this.model.core.oil.h, DX); }

  budget(): BudgetMsg {
    const b = this.model.core.budget;
    return { released: b.released, floating: this.floating(), evaporated: b.evaporated, dispersed: b.dispersed, stranded: b.stranded, left: b.left, held: this.heldByBooms };
  }

  /**
   * Surface drift on a 16 × 16 grid (every 16th cell, from the cell-8 offset), zero over land: [u, v] per node.
   * Only ever computed inside step(): reading it refreshes the engine's flow cache, so doing that whenever a frame is
   * drawn would make the run depend on the frame rate.
   */
  drift: Float32Array = new Float32Array(16 * 16 * 2);

  private computeDrift(): Float32Array {
    const a = this.model.arrows(), out = new Float32Array(16 * 16 * 2);
    for (let p = 0; p < a.length; p += 4) {
      const gi = Math.round((a[p] / DX + N / 2 - 0.5 - 8) / 16), gj = Math.round((a[p + 1] / DX + N / 2 - 0.5 - 8) / 16);
      out[2 * (gj * 16 + gi)] = a[p + 2];
      out[2 * (gj * 16 + gi) + 1] = a[p + 3];
    }
    return out;
  }
}

export interface LogEntry { chunk: number; edit: Edit }

/** Rebuild a run from its scenario and edit log up to `upTo` steps. Edits stamped with step c go in before step c runs. */
export function replay(sc: Scenario, log: LogEntry[], upTo: number, onStep?: (r: Runner) => void): Runner {
  const r = new Runner(sc);
  let e = 0;
  for (;;) {
    while (e < log.length && log[e].chunk === r.chunk) r.apply(log[e++].edit);
    if (r.chunk >= upTo) return r;
    r.step();
    onStep?.(r);
  }
}
