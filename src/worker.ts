// Runs the model off the main thread. Two roles, one file: the live run (load, edit, play, seek) and a determinism
// check that replays a log from scratch in a second copy of this worker and compares fingerprints step by step.
import { Runner, replay, type LogEntry } from './runner';
import { CHUNK, SCENARIOS, type Edit } from './world';
import type { FrameMsg, ToWorker } from './protocol';

let r: Runner | undefined;
let log: LogEntry[] = [];
let playing = false, speed = 300, debt = 0, last = 0, lastPost = 0, lastStats = 0;
let hashes: [number, string][] = [];
let marksDirty = true;
/** Which load or rewind this run is: the page drops anything from an older one still in flight. */
let gen = 0;

const post = (msg: FrameMsg | { type: string; [k: string]: unknown }, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(msg, transfer);

function frame(force = false) {
  if (!r) return;
  const now = performance.now();
  const o = r.model.core.oil, b = r.model.core.budget;
  const h = new Float32Array(o.h);
  const msg: FrameMsg = {
    type: 'frame', chunk: r.chunk, t: r.t, h, drift: r.drift, gen, hashes, playing,
    budget: { released: b.released, floating: r.floating(), evaporated: b.evaporated, dispersed: b.dispersed, stranded: b.stranded, left: b.left, held: r.heldByBooms },
  };
  hashes = [];
  if (marksDirty) { msg.marks = log.map((e) => ({ chunk: e.chunk, k: e.edit.k })); marksDirty = false; }
  if (force || now - lastStats > 400) { msg.stats = r.model.stats(); msg.warnings = r.model.warnings(); lastStats = now; }
  lastPost = now;
  post(msg, [h.buffer]);
}

function tick() {
  const now = performance.now();
  if (r && playing) {
    debt = Math.min(debt + ((now - last) / 1000) * (speed / CHUNK), 3);
    let stepped = false;
    while (debt >= 1 && performance.now() - now < 40) {
      r.step();
      hashes.push([r.chunk, r.fingerprint()]);
      debt--;
      stepped = true;
    }
    if (stepped && now - lastPost > 30) frame();
  }
  last = now;
  setTimeout(tick, 4);
}

function apply(edit: Edit) {
  if (!r) return;
  log.push({ chunk: r.chunk, edit });
  r.apply(edit);
  marksDirty = true;
  frame();
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.type === 'load') {
    gen = m.gen;
    r = new Runner(SCENARIOS.find((s) => s.id === m.scenario)!);
    log = []; hashes = [[0, r.fingerprint()]]; marksDirty = true; debt = 0;
    post({ type: 'scene', scene: r.scene, params: { ...r.model.params }, gen });
    frame(true);
  } else if (m.type === 'edit') apply(m.edit);
  else if (m.type === 'play') { playing = m.playing; speed = m.speed; debt = 0; frame(); }
  else if (m.type === 'step') { if (r) { r.step(); hashes.push([r.chunk, r.fingerprint()]); frame(true); } }
  else if (m.type === 'export' && r) post({ type: 'log', scenario: r.sc.id, log, chunk: r.chunk });
  else if (m.type === 'seek' && r) {
    // Branch: rebuild the run to that step from the log, and forget everything after it.
    const sc = r.sc, to = m.chunk;
    gen = m.gen;
    log = log.filter((x) => x.chunk <= to);
    r = replay(sc, log, to, (x) => { if (x.chunk % 10 === 0) post({ type: 'progress', done: x.chunk, of: to, gen }); });
    hashes = []; marksDirty = true; playing = false;
    post({ type: 'scene', scene: r.scene, params: { ...r.model.params }, gen });
    frame(true);
  } else if (m.type === 'verify') {
    const sc = SCENARIOS.find((s) => s.id === m.scenario)!, want = new Map(m.hashes);
    let checked = 0, first = -1;
    const fresh = new Runner(sc);
    if (want.has(0)) { checked++; if (want.get(0) !== fresh.fingerprint()) first = 0; }
    replay(sc, m.log, m.upTo, (x) => {
      const w = want.get(x.chunk);
      if (w !== undefined) { checked++; if (first < 0 && w !== x.fingerprint()) first = x.chunk; }
      if (x.chunk % 5 === 0) post({ type: 'progress', done: x.chunk, of: m.upTo });
    });
    post({ type: 'verified', checked, first });
  }
};

last = performance.now();
tick();
