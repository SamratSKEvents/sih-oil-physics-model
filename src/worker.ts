// Runs the model off the main thread, on the GPU when this browser has WebGPU, else on the CPU engine. Two roles,
// one file: the live run (load, edit, play, seek) and a determinism check that replays a log from scratch in a second
// copy of this worker, on the same engine, and compares fingerprints step by step.
import { Runner, type LogEntry } from './runner';
import { GpuSim, gpuDevice } from './gpu/sim';
import { CHUNK, SCENARIOS, type Edit, type Scenario } from './world';
import type { Engine, FrameMsg, ToWorker } from './protocol';

type Sim = Runner | GpuSim;

let r: Sim | undefined;
let engine: Engine = 'cpu';
let log: LogEntry[] = [];
let playing = false, speed = 300, debt = 0, last = 0, lastPost = 0, lastStats = 0;
let hashes: [number, string][] = [];
let marksDirty = true;
/** Which load or rewind this run is: the page drops anything from an older one still in flight. */
let gen = 0;

const post = (msg: FrameMsg | { type: string; [k: string]: unknown }, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(msg, transfer);

/** One thing at a time: a step, an edit or a rewind never overlaps another (GPU steps are asynchronous). */
let queue: Promise<unknown> = Promise.resolve();
const serial = <T,>(fn: () => Promise<T> | T): Promise<T> => {
  const next = queue.then(fn);
  queue = next.catch((e) => console.error(e));
  return next;
};

async function make(sc: Scenario, want: Engine): Promise<Sim> {
  const device = want === 'gpu' ? await gpuDevice() : null;
  engine = device ? 'gpu' : 'cpu';
  return device ? GpuSim.create(device, sc) : new Runner(sc);
}

const params = (s: Sim) => (s instanceof GpuSim ? s.params : s.model.params) as Record<string, number>;

async function stepOnce(s: Sim) {
  await s.step();
  if (s instanceof Runner) hashes.push([s.chunk, s.fingerprint()]);
  else hashes.push(...s.takeHashes());
}

/** Rebuild a run from its scenario and edit log up to `upTo` steps. Edits stamped with step c go in before step c. */
async function replay(sc: Scenario, want: Engine, list: LogEntry[], upTo: number, onStep?: (s: Sim, fresh: [number, string][]) => void) {
  const s = await make(sc, want);
  let e = 0;
  for (;;) {
    while (e < list.length && list[e].chunk === s.chunk) await s.apply(list[e++].edit);
    if (s.chunk >= upTo) break;
    await s.step();
    onStep?.(s, s instanceof Runner ? [[s.chunk, s.fingerprint()]] : s.takeHashes());
  }
  if (s instanceof GpuSim) { await s.drain(); onStep?.(s, s.takeHashes()); }
  return s;
}

async function frame(force = false) {
  const s = r;
  if (!s) return;
  const now = performance.now();
  const h = s instanceof GpuSim ? await s.thickness() : new Float32Array(s.model.core.oil.h);
  if (s !== r) return;
  const msg: FrameMsg = { type: 'frame', gen, engine, chunk: s.chunk, t: s.t, h, drift: s.drift, hashes, playing, budget: s.budget() };
  hashes = [];
  if (marksDirty) { msg.marks = log.map((e) => ({ chunk: e.chunk, k: e.edit.k })); marksDirty = false; }
  if (force || now - lastStats > 400) {
    msg.stats = s instanceof GpuSim ? s.statsRows(h) : s.model.stats();
    msg.warnings = s.warnings();
    lastStats = now;
  }
  lastPost = now;
  post(msg, [h.buffer]);
}

/** The steps due at the chosen speed, for up to 40 ms, then a frame when one is due. */
async function tick() {
  const now = performance.now();
  if (r && playing) {
    debt = Math.min(debt + ((now - last) / 1000) * (speed / CHUNK), 64);
    let stepped = false;
    while (debt >= 1 && performance.now() - now < 40 && playing) {
      await stepOnce(r);
      debt--;
      stepped = true;
    }
    if (stepped && performance.now() - lastPost > 30) await frame();
  }
  last = now;
}
const loop = () => { void serial(tick).finally(() => setTimeout(loop, 4)); };

async function apply(edit: Edit) {
  if (!r) return;
  log.push({ chunk: r.chunk, edit });
  await r.apply(edit);
  marksDirty = true;
  await frame();
}

const sceneMsg = (s: Sim) => post({ type: 'scene', scene: s.scene, params: { ...params(s) }, gen, engine });

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.type === 'load') void serial(async () => {
    if (r instanceof GpuSim) r.destroy();
    r = undefined;
    gen = m.gen;
    const s = await make(SCENARIOS.find((x) => x.id === m.scenario)!, m.engine);
    r = s;
    log = []; hashes = [[0, s.fingerprint()]]; marksDirty = true; debt = 0;
    sceneMsg(s);
    await frame(true);
  });
  else if (m.type === 'edit') void serial(() => apply(m.edit));
  else if (m.type === 'play') {
    playing = m.playing; speed = m.speed; debt = 0;
    // paused: finish the GPU steps in flight so the fingerprint shown is the current state's
    void serial(async () => { if (!playing && r instanceof GpuSim) { await r.drain(); hashes.push(...r.takeHashes()); } await frame(); });
  }
  else if (m.type === 'step') void serial(async () => {
    if (!r) return;
    await stepOnce(r);
    if (r instanceof GpuSim) { await r.drain(); hashes.push(...r.takeHashes()); }
    await frame(true);
  });
  else if (m.type === 'export') void serial(async () => {
    if (!r) return;
    if (r instanceof GpuSim) { await r.drain(); hashes.push(...r.takeHashes()); await frame(); }
    post({ type: 'log', scenario: r.sc.id, engine, log, chunk: r.chunk });
  });
  else if (m.type === 'seek') void serial(async () => {
    if (!r) return;
    // Branch: rebuild the run to that step from the log, and forget everything after it.
    const sc = r.sc, to = m.chunk;
    if (r instanceof GpuSim) r.destroy();
    r = undefined;
    gen = m.gen;
    log = log.filter((x) => x.chunk <= to);
    const s = await replay(sc, engine, log, to, (x) => { if (x.chunk % 10 === 0) post({ type: 'progress', done: x.chunk, of: to, gen }); });
    r = s;
    hashes = []; marksDirty = true; playing = false;
    sceneMsg(s);
    await frame(true);
  });
  else if (m.type === 'verify') void serial(async () => {
    const sc = SCENARIOS.find((x) => x.id === m.scenario)!, want = new Map(m.hashes);
    let checked = 0, first = -1;
    const check = (c: number, h: string) => {
      const w = want.get(c);
      if (w === undefined) return;
      checked++;
      if (first < 0 && w !== h) first = c;
    };
    // step 0: the state after setup, before any step
    const s0 = await make(sc, m.engine);
    check(0, s0.fingerprint());
    if (s0 instanceof GpuSim) s0.destroy();
    const s = await replay(sc, m.engine, m.log, m.upTo, (x, fresh) => {
      for (const [c, h] of fresh) check(c, h);
      if (x.chunk % 5 === 0) post({ type: 'progress', done: x.chunk, of: m.upTo });
    });
    if (s instanceof GpuSim) s.destroy();
    post({ type: 'verified', checked, first, engine });
  });
};

last = performance.now();
loop();
