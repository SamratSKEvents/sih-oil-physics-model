// The page: scenario picker, drawing tools, live conditions, timeline and the determinism check. The simulation runs
// in a worker; this side keeps the scene a person is editing and sends every change to it as a logged edit.
import './style.css';
import { View, BONN, type Overlay } from './render';
import { TimeChart } from './charts';
import { stroke, strokeAt } from './engine/film/shapes';
import { CHUNK, DX, HALF, N, SCENARIOS, cellCentre, landMask, ring, spiral, type Pt, type Scenario, type Scene, type Stroke } from './world';
import type { Engine, FrameMsg, FromWorker, ToWorker } from './protocol';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

/* ------------------------------------------------------------------ icons */
const svg = (d: string) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;
const ICON = {
  select: svg('<path d="M4 3l7 17 2-7 7-2z"/>'),
  boom: svg('<path d="M3 17c4-6 14-6 18 0"/><circle cx="7" cy="12.6" r="1.4"/><circle cx="12" cy="11" r="1.4"/><circle cx="17" cy="12.6" r="1.4"/>'),
  wind: svg('<path d="M3 8h10a3 3 0 1 0-3-3"/><path d="M3 12h15a3 3 0 1 1-3 3"/><path d="M3 16h7"/>'),
  current: svg('<path d="M2 8c2.5-2 4.5-2 7 0s4.5 2 7 0 4.5-2 6 0"/><path d="M2 15c2.5-2 4.5-2 7 0s4.5 2 7 0 4.5-2 6 0"/>'),
  spiral: svg('<path d="M12 12a2 2 0 1 1 2 2 4 4 0 1 1-4-4 6 6 0 1 1-6 6"/>'),
  eddy: svg('<path d="M20 12a8 8 0 1 1-3-6.2"/><path d="M17 2v4h-4"/>'),
  spill: svg('<path d="M12 3c3 4.5 6 7.6 6 11a6 6 0 0 1-12 0c0-3.4 3-6.5 6-11z"/>'),
  source: svg('<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="7" opacity=".6"/><circle cx="12" cy="12" r="10.5" opacity=".3"/>'),
  play: svg('<path d="M7 4l13 8-13 8z" fill="currentColor"/>'),
  pause: svg('<path d="M7 4h3v16H7zM14 4h3v16h-3z" fill="currentColor"/>'),
  step: svg('<path d="M5 4l10 8-10 8z"/><path d="M19 4v16"/>'),
  restart: svg('<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 3v6h6"/>'),
};

type Tool = keyof typeof TOOLS;
const TOOLS = {
  select: ['Move', 'Drag anything you placed. Select it to tune it; Delete removes it.'],
  boom: ['Boom', 'Drag to lay a floating boom. Oil cannot cross it.'],
  wind: ['Wind', 'Drag a path: the wind follows it from where you start to where you let go.'],
  current: ['Current', 'Drag a path: the surface current follows it.'],
  spiral: ['Spiral', 'Drag from the centre outwards: a cyclonic wind spiralling in.'],
  eddy: ['Eddy', 'Drag from the centre outwards: a rotating current eddy.'],
  spill: ['Spill', 'Click, or drag for a wider slick, to release oil at once.'],
  source: ['Leak', 'Click to place a leak that keeps releasing oil.'],
} as const;

const PARAMS: { key: string; label: string; unit: string; min: number; max: number; step: number }[] = [
  { key: 'windSpeed', label: 'Wind speed', unit: 'm/s', min: 0, max: 25, step: 0.5 },
  { key: 'windDirDeg', label: 'Wind towards', unit: '°', min: 0, max: 359, step: 1 },
  { key: 'driftU', label: 'Background current east', unit: 'm/s', min: -0.6, max: 0.6, step: 0.01 },
  { key: 'driftV', label: 'Background current north', unit: 'm/s', min: -0.6, max: 0.6, step: 0.01 },
  { key: 'uTide', label: 'Tidal current', unit: 'm/s', min: 0, max: 1, step: 0.01 },
  { key: 'eddySpeed', label: 'Turbulent eddies', unit: 'm/s', min: 0, max: 0.2, step: 0.005 },
  { key: 'windage', label: 'Windage', unit: '× wind', min: 0, max: 0.05, step: 0.001 },
  { key: 'Kh', label: 'Turbulent diffusion', unit: 'm²/s', min: 0, max: 10, step: 0.1 },
  { key: 'hTerminalUm', label: 'Film breaks up below', unit: 'µm', min: 5, max: 200, step: 1 },
];
/** Simulated seconds per wall second; Infinity runs as fast as the machine allows. */
const SPEEDS = [300, 1200, 6000, Infinity];

/* ------------------------------------------------------------------ state */
const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
const send = (m: ToWorker) => worker.postMessage(m);
const view = new View($<HTMLCanvasElement>('view'));

let sc: Scenario = SCENARIOS[0];
let scene: Scene = { booms: [], strokes: [], sources: [] };
let params: Record<string, number> = {};
let tool: Tool = 'select';
let selected: Overlay['selected'] = null;
let draft: Overlay['draft'] = null;
let playing = false, speed = 600;
let live: FrameMsg | null = null;
let snaps: { chunk: number; h: Float32Array }[] = [];
let snapEvery = 10;
let hashes = new Map<number, string>();
let marks: NonNullable<FrameMsg['marks']> = [];
let maxChunk = 0;
let history: number | null = null; // the step being viewed from the recording, or null when live
let busy: { what: string; done: number; of: number } | null = null;
let nextId = 100;
let spillVolume = 300, leakRate = 40;
let rate = 0, rateT = 0, rateSim = 0;
let gen = 0;
/** The newest fingerprint known: on the GPU a step's arrives two steps after it runs. */
let latestHash: [number, string] = [-1, ''];
/** The engine asked for (remembered per browser) and the one actually running (WebGPU can be missing). */
let enginePref: Engine = (() => { try { return localStorage.getItem('engine') === 'cpu' ? 'cpu' : 'gpu'; } catch { return 'gpu'; } })();
let engineNow: Engine = enginePref;

/* ------------------------------------------------------------------ worker */
worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const m = e.data;
  if ('gen' in m && m.gen !== undefined && m.gen !== gen) return; // from a run since replaced
  if (m.type === 'frame') {
    if (m.engine !== engineNow) { engineNow = m.engine; showEngine(); }
    for (const [c, h] of m.hashes) { hashes.set(c, h); if (c >= latestHash[0]) latestHash = [c, h]; }
    if (m.marks) marks = m.marks;
    if (m.stats) showStats(m.stats, m.warnings ?? []);
    const now = performance.now();
    if (live && m.t > rateSim && now > rateT) rate += 0.3 * ((m.t - rateSim) / ((now - rateT) / 1000) - rate);
    rateT = now; rateSim = m.t;
    const lastSnap = snaps.at(-1);
    if (!lastSnap || m.chunk >= lastSnap.chunk + snapEvery) snaps.push({ chunk: m.chunk, h: m.h });
    // Long fast runs: keep the recording under ~100 MB by halving its density instead of growing without end.
    if (snaps.length > 400) { snaps = snaps.filter((_, i) => i % 2 === 0); snapEvery *= 2; }
    maxChunk = Math.max(maxChunk, m.chunk);
    record(m);
    live = m;
    playing = m.playing;
    busy = null;
    showBudget();
    showTime();
  } else if (m.type === 'scene') {
    scene = m.scene; params = m.params;
    nextId = Math.max(nextId, ...scene.booms.map((b) => b.id + 1), ...scene.strokes.map((s) => s.id + 1), ...scene.sources.map((s) => s.id + 1));
    selected = null;
    buildParams(); buildSelected();
  } else if (m.type === 'progress') {
    if (busy) { busy.done = m.done; busy.of = m.of; }
  } else if (m.type === 'log') verify(m.scenario, m.engine, m.log, m.chunk);
};

function load(next: Scenario) {
  sc = next;
  snaps = []; snapEvery = 10; samples = []; sampleEvery = 1; hashes = new Map(); latestHash = [-1, '']; marks = []; maxChunk = 0; history = null; live = null; selected = null; rate = 0;
  view.setLand(landMask(sc));
  send({ type: 'load', scenario: sc.id, gen: ++gen, engine: enginePref });
  $('verify-out').textContent = '';
  buildScenarios();
  setPlaying(true);
}

function setPlaying(p: boolean) {
  if (p && history !== null) return resume();
  playing = p;
  send({ type: 'play', playing, speed });
  $('play').innerHTML = playing ? ICON.pause : ICON.play;
}

function pushScene() {
  if (history !== null) return;
  send({ type: 'edit', edit: { k: 'scene', scene } });
}

/** Leave the recording and carry on from the step being viewed; everything after it is dropped. */
function resume() {
  if (history === null) return;
  const c = history;
  history = null;
  snaps = snaps.filter((s) => s.chunk <= c);
  samples = samples.filter((s) => s.t <= c * CHUNK);
  for (const k of [...hashes.keys()]) if (k > c) hashes.delete(k);
  latestHash = [-1, ''];
  maxChunk = c;
  busy = { what: 'Replaying the log to this point', done: 0, of: c };
  send({ type: 'seek', chunk: c, gen: ++gen });
  $('verify-out').textContent = '';
}

/* ------------------------------------------------------------------ determinism */
let verifier: Worker | undefined;
$('verify').onclick = () => { if (!busy && history === null) send({ type: 'export' }); };

function verify(scenario: string, engine: Engine, log: import('./runner').LogEntry[], upTo: number) {
  verifier?.terminate();
  verifier = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  const out = $('verify-out'), btn = $<HTMLButtonElement>('verify');
  const want = [...hashes].filter(([c]) => c <= upTo);
  btn.disabled = true;
  out.innerHTML = `Replaying ${upTo} steps and ${log.length} edits from zero in a second worker, on the ${engine === 'gpu' ? 'GPU' : 'CPU'}… <div class="bar"><i style="width:0%"></i></div>`;
  verifier.onmessage = (e: MessageEvent<FromWorker>) => {
    const m = e.data;
    if (m.type === 'progress') out.querySelector<HTMLElement>('.bar i')!.style.width = `${(100 * m.done) / Math.max(1, m.of)}%`;
    if (m.type === 'verified') {
      btn.disabled = false;
      out.innerHTML = m.first < 0
        ? `<span class="ok">✓ Bit-identical</span><br>${m.checked} of ${m.checked} steps match: all 65,536 cells of thickness and momentum, as raw ${m.engine === 'gpu' ? '32-bit floats on this GPU' : '64-bit floats'}, after replaying ${log.length} edit${log.length === 1 ? '' : 's'}.`
        : `<span class="bad">✗ Diverged at step ${m.first}</span><br>${m.checked} steps checked.`;
      verifier?.terminate(); verifier = undefined;
    }
  };
  verifier.postMessage({ type: 'verify', scenario, engine, log, upTo, hashes: want } satisfies ToWorker);
}

/* ------------------------------------------------------------------ engine */
function showEngine() {
  const b = $('engine');
  b.innerHTML = engineNow === 'gpu' ? 'Engine <code>GPU</code>' : 'Engine <code>CPU</code>';
  b.title = engineNow === 'gpu'
    ? 'Running on WebGPU compute shaders (32-bit). Click to run on the CPU engine (64-bit) instead.'
    : enginePref === 'gpu' ? 'This browser has no WebGPU, so the CPU engine runs.' : 'Running the CPU engine (64-bit). Click to run on the GPU instead.';
}
$('engine').onclick = () => {
  enginePref = engineNow === 'gpu' ? 'cpu' : 'gpu';
  try { localStorage.setItem('engine', enginePref); } catch { /* per-browser convenience only */ }
  engineNow = enginePref;
  showEngine();
  load(sc);
};

/* ------------------------------------------------------------------ panels */
function buildScenarios() {
  $('scenarios').replaceChildren(...SCENARIOS.map((s) => {
    const b = document.createElement('button');
    b.className = s === sc ? 'on' : '';
    b.innerHTML = `<b>${s.name}</b><span>${s.blurb}</span>`;
    b.onclick = () => load(s);
    return b;
  }));
}

function buildTools() {
  $('tools').replaceChildren(...(Object.keys(TOOLS) as Tool[]).map((t) => {
    const b = document.createElement('button');
    b.className = t === tool ? 'on' : '';
    b.innerHTML = `${ICON[t]}${TOOLS[t][0]}`;
    b.title = TOOLS[t][1];
    b.onclick = () => setTool(t);
    return b;
  }));
  $('tool-hint').textContent = TOOLS[tool][1];
  const opts = $('tool-options');
  opts.replaceChildren();
  if (tool === 'spill') opts.append(slider('Volume', 'm³', 20, 2000, 10, spillVolume, (v) => { spillVolume = v; }));
  if (tool === 'source') opts.append(slider('Leak rate', 'm³/h', 5, 300, 5, leakRate, (v) => { leakRate = v; }));
}

function setTool(t: Tool) {
  tool = t;
  draft = null;
  buildTools();
}

function slider(label: string, unit: string, min: number, max: number, step: number, value: number, set: (v: number) => void, fmt = (v: number) => `${+v.toFixed(3)}`) {
  const f = document.createElement('div');
  f.className = 'field';
  f.innerHTML = `<label><span>${label}</span><output>${fmt(value)} ${unit}</output></label><input type="range" min="${min}" max="${max}" step="${step}" value="${value}">`;
  const input = f.querySelector('input')!, out = f.querySelector('output')!;
  input.oninput = () => { const v = +input.value; out.textContent = `${fmt(v)} ${unit}`; set(v); };
  return f;
}

function buildParams() {
  $('params').replaceChildren(...PARAMS.map((p) => slider(p.label, p.unit, p.min, p.max, p.step, params[p.key] ?? 0, (v) => {
    params[p.key] = v;
    if (history === null) send({ type: 'edit', edit: { k: 'param', key: p.key, value: v } });
  })));
}

function find(sel: NonNullable<Overlay['selected']>) {
  return sel.kind === 'boom' ? scene.booms.find((b) => b.id === sel.id) : sel.kind === 'stroke' ? scene.strokes.find((s) => s.id === sel.id) : scene.sources.find((s) => s.id === sel.id);
}

function buildSelected() {
  const sec = $('selected-sec'), box = $('selected');
  const obj = selected && find(selected);
  sec.hidden = !obj;
  if (!selected || !obj) return;
  box.replaceChildren();
  const title = (text: string, color: string) => { const t = document.createElement('div'); t.className = 'obj-title'; t.innerHTML = `<i style="background:${color}"></i>${text}`; box.append(t); };
  const buttons: [string, () => void, string?][] = [];
  if (selected.kind === 'stroke') {
    const s = obj as Stroke, wind = s.kind === 'wind';
    title(s.label, wind ? 'var(--wind)' : 'var(--current)');
    box.append(
      slider('Speed', 'm/s', 0, wind ? 30 : 1.5, wind ? 0.5 : 0.01, s.speed, (v) => { s.speed = v; pushScene(); }),
      slider('Width', 'm', 150, 9000, 50, s.width, (v) => { s.width = v; pushScene(); }, (v) => v.toFixed(0)),
    );
    buttons.push(['Reverse', () => { s.path.reverse(); pushScene(); }]);
  } else if (selected.kind === 'source') {
    const s = obj as Scene['sources'][number];
    title('Leak', 'var(--source)');
    box.append(slider('Leak rate', 'm³/h', 0, 300, 5, s.rate, (v) => { s.rate = v; pushScene(); }));
  } else title('Boom', 'var(--boom)');
  buttons.push(['Delete', remove, 'danger']);
  const row = document.createElement('div');
  row.className = 'btns';
  for (const [text, fn, cls] of buttons) { const b = document.createElement('button'); b.textContent = text; b.onclick = fn; if (cls) b.className = cls; row.append(b); }
  box.append(row);
}

function remove() {
  if (!selected || history !== null) return;
  const { kind, id } = selected;
  if (kind === 'boom') scene.booms = scene.booms.filter((b) => b.id !== id);
  if (kind === 'stroke') scene.strokes = scene.strokes.filter((s) => s.id !== id);
  if (kind === 'source') scene.sources = scene.sources.filter((s) => s.id !== id);
  selected = null;
  pushScene();
  buildSelected();
}

const BUDGET_ROWS: [keyof FrameMsg['budget'], string, string][] = [
  ['floating', 'Floating', '#eb6834'],
  ['evaporated', 'Evaporated', '#2a78d6'],
  ['dispersed', 'Dispersed into water', '#1baf7a'],
  ['stranded', 'Stranded / at booms', '#eda100'],
  ['left', 'Left the area', '#e87ba4'],
];
const m3 = (v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(1));

function showBudget() {
  if (!live) return;
  const b = live.budget;
  $('budget').innerHTML = `<dt>Released</dt><dd>${m3(b.released)} m³</dd>` + BUDGET_ROWS.map(([k, name, c]) => `<dt><i style="background:${c}"></i>${name}</dt><dd>${m3(Math.max(0, b[k]))} m³</dd>`).join('');
}

function showStats(rows: [string, string][], warnings: string[]) {
  $('stats').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  $('warnings').innerHTML = warnings.map((w) => `<div class="warn">⚠ ${w}</div>`).join('');
}

const clock = (chunk: number) => { const min = (chunk * CHUNK) / 60; return `T+${String(Math.floor(min / 60)).padStart(2, '0')}:${String(Math.floor(min % 60)).padStart(2, '0')}`; };

function showTime() {
  const c = history ?? live?.chunk ?? 0;
  $('clock').textContent = clock(c);
  $('rate').textContent = history !== null ? 'recording' : playing ? `${Math.round(rate)}× real time` : 'paused';
  $('step').textContent = String(c);
  const fp = hashes.get(c);
  $('fp').textContent = fp ?? (history === null && latestHash[1] ? latestHash[1] : '········');
  $('fp').parentElement!.title = fp || history !== null ? `Fingerprint of every cell of the oil field at step ${c}` : `Fingerprint at step ${latestHash[0]}; the newest steps' arrive a moment later`;
}


/* ------------------------------------------------------------------ live charts */
/** One row per recorded minute: budget terms, sheen area and thickest oil. Thinned like the recording on long runs. */
interface Sample { t: number; b: FrameMsg['budget']; area: number; thick: number }
let samples: Sample[] = [];
let sampleEvery = 1;
const hm = (t: number) => `${Math.floor(t / 3600)}:${String(Math.floor((t % 3600) / 60)).padStart(2, '0')}`;
const num = (v: number) => (v === 0 ? '0' : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
const chartBudget = new TimeChart($('chart-budget'), { series: BUDGET_ROWS.map(([key, label, color]) => ({ key, label, color })), stacked: true, unit: 'm³', fmt: num, xfmt: hm });
const chartArea = new TimeChart($('chart-area'), { series: [{ key: 'area', label: 'Slick area', color: '#2a78d6' }], unit: 'km²', fmt: num, xfmt: hm });
const chartThick = new TimeChart($('chart-thick'), { series: [{ key: 'thick', label: 'Thickest oil', color: '#eb6834' }], unit: 'µm', fmt: num, xfmt: hm });
let chartsAt = 0;

function record(m: FrameMsg) {
  const last = samples.at(-1);
  if (last && m.t < last.t + sampleEvery * CHUNK) return;
  let area = 0, thick = 0;
  for (let k = 0; k < m.h.length; k++) { const v = m.h[k]; if (v >= 0.04e-6) area++; if (v > thick) thick = v; }
  samples.push({ t: m.t, b: { ...m.budget }, area: (area * DX * DX) / 1e6, thick: thick * 1e6 });
  if (samples.length > 600) { samples = samples.filter((_, i) => i % 2 === 0); sampleEvery *= 2; }
}

function showCharts(now: number) {
  if (now - chartsAt < 250 || samples.length < 2) return;
  chartsAt = now;
  const xs = samples.map((s) => s.t);
  chartBudget.set(xs, BUDGET_ROWS.map(([k]) => samples.map((s) => Math.max(0, s.b[k]))));
  chartArea.set(xs, [samples.map((s) => s.area)]);
  chartThick.set(xs, [samples.map((s) => s.thick)]);
}

/* ------------------------------------------------------------------ wind and current arrows */
const show = { wind: false, current: false, tracers: true };
for (const k of ['wind', 'current', 'tracers'] as const) {
  const box = $<HTMLInputElement>(`show-${k}`);
  box.checked = show[k];
  box.onchange = () => { show[k] = box.checked; };
}

/** Wind at the 16 × 16 arrow nodes: the global wind, bent by drawn wind paths (the engine's localWind rule). */
function windField(): Float32Array {
  const out = new Float32Array(512), th = ((params.windDirDeg ?? 0) * Math.PI) / 180, W = params.windSpeed ?? 0;
  const paths = scene.strokes.filter((s) => s.kind === 'wind').map((s) => ({ s, r: stroke(s.id, 'wind', s.path.flat(), s.width, s.speed) }));
  const at = { w: 0, ux: 0, uy: 0 };
  for (let g = 0; g < 256; g++) {
    const x = cellCentre(8 + 16 * (g % 16)), y = cellCentre(8 + 16 * Math.floor(g / 16));
    let wx = W * Math.sin(th), wy = W * Math.cos(th);
    for (const { s, r } of paths) {
      if (strokeAt(r, x, y, at).w <= 0) continue;
      wx += at.w * (s.speed * at.ux - wx); wy += at.w * (s.speed * at.uy - wy);
    }
    out[2 * g] = wx; out[2 * g + 1] = wy;
  }
  return out;
}

/** Current: the surface drift the oil follows, less its windage share of the wind. Water, tide, eddies and drawn currents. */
function currentField(drift: Float32Array, wind: Float32Array): Float32Array {
  const out = new Float32Array(512), a = params.windage ?? 0;
  for (let i = 0; i < 512; i++) out[i] = drift[i] === 0 && drift[i ^ 1] === 0 ? 0 : drift[i] - a * wind[i];
  return out;
}

/* ------------------------------------------------------------------ timeline */
function buildTimeline() {
  $('restart').innerHTML = ICON.restart;
  $('stepbtn').innerHTML = ICON.step;
  $('play').innerHTML = playing ? ICON.pause : ICON.play;
  $('play').onclick = () => setPlaying(!playing || history !== null);
  $('stepbtn').onclick = () => { if (history === null) { setPlaying(false); send({ type: 'step' }); } };
  $('restart').onclick = () => load(sc);
  $('speeds').replaceChildren(...SPEEDS.map((s) => {
    const b = document.createElement('button');
    b.textContent = s === Infinity ? 'Max' : `${s}×`;
    b.title = s === Infinity ? 'As fast as this machine can run it' : `${s / 60} simulated minutes per second`;
    b.className = s === speed ? 'on' : '';
    b.onclick = () => { speed = s; if (playing) send({ type: 'play', playing, speed }); buildTimeline(); };
    return b;
  }));
  const track = $('track');
  const scrub = (e: PointerEvent) => {
    if (!snaps.length || busy) return;
    const r = track.getBoundingClientRect(), span = Math.max(maxChunk, 1);
    const want = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * span;
    const snap = snaps.reduce((a, s) => (Math.abs(s.chunk - want) < Math.abs(a.chunk - want) ? s : a), snaps[0]);
    if (snap.chunk >= (live?.chunk ?? 0)) { history = null; return; }
    if (playing) setPlaying(false);
    history = snap.chunk;
    selected = null; buildSelected();
  };
  track.onpointerdown = (e) => { track.setPointerCapture(e.pointerId); scrub(e); };
  track.onpointermove = (e) => { if (track.hasPointerCapture(e.pointerId)) scrub(e); };
}

function drawTimeline() {
  const span = Math.max(maxChunk, 1), at = history ?? live?.chunk ?? 0;
  $('fill').style.width = `${(100 * (live?.chunk ?? 0)) / span}%`;
  $('head').style.left = `calc(${(100 * at) / span}% - 1px)`;
  const colours = { scene: '#fbbf24', param: '#38bdf8', spill: '#f59e0b' };
  const key = marks.length + ':' + span;
  const box = $('marks');
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    const seen = new Set<string>();
    box.innerHTML = marks.filter((m) => { const k = `${m.k}${Math.round((400 * m.chunk) / span)}`; if (seen.has(k)) return false; seen.add(k); return true; })
      .map((m) => `<i style="left:${(100 * m.chunk) / span}%;background:${colours[m.k]}" title="${m.k} at ${clock(m.chunk)}"></i>`).join('');
  }
  $('tl-time').textContent = history !== null ? `${clock(history)} of ${clock(maxChunk)}` : `${(((live?.chunk ?? 0) * CHUNK) / 3600).toFixed(1)} h simulated`;
}

function drawBanner() {
  const b = $('banner');
  if (busy) {
    b.hidden = false;
    const html = `${busy.what}… <div class="bar"><i style="width:${(100 * busy.done) / Math.max(1, busy.of)}%"></i></div>`;
    if (b.innerHTML !== html) b.innerHTML = html;
    return;
  }
  if (history === null) { b.hidden = true; b.dataset.at = ''; return; }
  b.hidden = false;
  if (b.dataset.at === String(history)) return;
  b.dataset.at = String(history);
  b.innerHTML = `Viewing the recording at <b>${clock(history)}</b> <button class="primary">Resume from here</button><button>Back to live</button>`;
  const [go, back] = b.querySelectorAll('button');
  go.onclick = () => resume();
  back.onclick = () => { history = null; };
}

/* ------------------------------------------------------------------ canvas input */
const canvas = view.canvas;
let drag: { kind: 'move'; last: Pt } | { kind: 'draw' } | null = null;

const dist = (p: Pt, a: Pt, b: Pt) => {
  const bx = b[0] - a[0], by = b[1] - a[1], L = bx * bx + by * by;
  const t = L ? Math.max(0, Math.min(1, ((p[0] - a[0]) * bx + (p[1] - a[1]) * by) / L)) : 0;
  return Math.hypot(p[0] - a[0] - t * bx, p[1] - a[1] - t * by);
};
const nearPath = (p: Pt, path: Pt[]) => path.reduce((d, _, i) => (i ? Math.min(d, dist(p, view.px(path[i - 1]), view.px(path[i]))) : d), Infinity);

function hit(px: Pt): Overlay['selected'] {
  for (const s of scene.sources) if (Math.hypot(...(view.px(s.at).map((v, i) => v - px[i]) as Pt)) < 14) return { kind: 'source', id: s.id };
  for (const b of scene.booms) if (nearPath(px, b.path) < 10) return { kind: 'boom', id: b.id };
  for (const s of [...scene.strokes].reverse()) if (nearPath(px, s.path) < 10) return { kind: 'stroke', id: s.id };
  return null;
}

const local = (e: PointerEvent): Pt => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
const clampM = ([x, y]: Pt): Pt => [Math.max(-HALF, Math.min(HALF, x)), Math.max(-HALF, Math.min(HALF, y))];

let pendingScene = false;
const pushSoon = () => { if (!pendingScene) { pendingScene = true; requestAnimationFrame(() => { pendingScene = false; pushScene(); }); } };

canvas.onpointerdown = (e) => {
  if (history !== null || busy) return;
  canvas.setPointerCapture(e.pointerId);
  const p = local(e), at = clampM(view.m(...p));
  if (tool === 'select') {
    selected = hit(p);
    buildSelected();
    if (selected) drag = { kind: 'move', last: at };
    return;
  }
  if (tool === 'source') {
    scene.sources.push({ id: nextId++, at, rate: leakRate });
    selected = { kind: 'source', id: nextId - 1 };
    pushScene(); buildSelected();
    return;
  }
  draft = { tool, path: [at], radius: ['spiral', 'eddy', 'spill'].includes(tool) ? 0 : undefined };
  drag = { kind: 'draw' };
};

canvas.onpointermove = (e) => {
  const p = local(e);
  canvas.style.cursor = tool === 'select' ? (hit(p) ? 'grab' : 'default') : 'crosshair';
  if (!drag) return;
  const at = clampM(view.m(...p));
  if (drag.kind === 'move' && selected) {
    const dx = at[0] - drag.last[0], dy = at[1] - drag.last[1], obj = find(selected);
    drag.last = at;
    if (!obj) return;
    if ('path' in obj) obj.path = obj.path.map(([x, y]) => [x + dx, y + dy] as Pt);
    else obj.at = [obj.at[0] + dx, obj.at[1] + dy];
    pushSoon();
  } else if (drag.kind === 'draw' && draft) {
    if (draft.radius !== undefined) draft.radius = Math.hypot(at[0] - draft.path[0][0], at[1] - draft.path[0][1]);
    else if (Math.hypot(...(view.px(draft.path.at(-1)!).map((v, i) => v - p[i]) as Pt)) > 7) draft.path.push(at);
  }
};

canvas.onpointerup = () => {
  const d = draft;
  drag = null; draft = null;
  if (!d) return;
  const pathLen = d.path.reduce((s, q, i) => (i ? s + Math.hypot(q[0] - d.path[i - 1][0], q[1] - d.path[i - 1][1]) : 0), 0);
  let added: Overlay['selected'] = null;
  if (d.tool === 'spill') {
    send({ type: 'edit', edit: { k: 'spill', spill: { at: d.path[0], radius: Math.max(250, d.radius ?? 0), volume: spillVolume } } });
    return;
  }
  if (d.tool === 'spiral' || d.tool === 'eddy') {
    const R = Math.max(600, d.radius ?? 0), spin = d.tool === 'spiral';
    scene.strokes.push({ id: nextId, kind: spin ? 'wind' : 'current', label: spin ? 'Spiral wind' : 'Eddy', width: spin ? R * 0.32 : R * 0.9, speed: spin ? 15 : 0.45, path: spin ? spiral(d.path[0], R) : ring(d.path[0], R) });
    added = { kind: 'stroke', id: nextId++ };
  } else if (d.path.length >= 2 && pathLen > 150) {
    if (d.tool === 'boom') { scene.booms.push({ id: nextId, path: d.path }); added = { kind: 'boom', id: nextId++ }; }
    else {
      const wind = d.tool === 'wind';
      scene.strokes.push({ id: nextId, kind: wind ? 'wind' : 'current', label: wind ? 'Wind path' : 'Current path', width: Math.round(Math.min(3000, Math.max(400, pathLen / 4)) / 50) * 50, speed: wind ? 14 : 0.4, path: d.path });
      added = { kind: 'stroke', id: nextId++ };
    }
  }
  if (added) { selected = added; pushScene(); buildSelected(); }
};

addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.key === 'Delete' || e.key === 'Backspace') remove();
  else if (e.key === ' ') { e.preventDefault(); setPlaying(!playing || history !== null); }
  else if (e.key === 'Escape') setTool('select');
});

/* ------------------------------------------------------------------ frame loop */
$('legend').innerHTML = BONN.map((b) => `<div><i style="background:${b.css}"></i>${b.name} ≥ ${b.min * 1e6 >= 1 ? b.min * 1e6 : (b.min * 1e6).toFixed(2)} µm</div>`).join('')
  + `<div><i style="background:#fbbf24"></i>Boom</div><div><i style="background:#7dd3fc"></i>Wind path</div><div><i style="background:#6ee7b7"></i>Current path</div>`
  + `<div><i style="background:#9fb3c8"></i>Wind arrow</div><div><i style="background:#5ef0b0"></i>Current arrow</div>`;

let lastT = performance.now();
function frame(now: number) {
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;
  view.resize();
  const kmPx = view.len(1000);
  $('scale').querySelector('i')!.style.width = `${kmPx}px`;
  const shown = history !== null ? snaps.find((s) => s.chunk === history)?.h ?? null : live?.h ?? null;
  const wind = show.wind || show.current ? windField() : null;
  const arrows = {
    wind: show.wind ? wind : null,
    current: show.current && live ? currentField(live.drift, wind!) : null,
  };
  view.draw(shown, show.tracers ? live?.drift ?? null : null, history === null && playing ? (speed === Infinity ? rate : speed) : 0, dt, { scene, selected, draft, time: now / 1000, arrows });
  showCharts(now);
  drawTimeline();
  drawBanner();
  showTime();
  requestAnimationFrame(frame);
}

showEngine();
buildTools();
buildTimeline();
load(SCENARIOS[0]);
requestAnimationFrame(frame);
