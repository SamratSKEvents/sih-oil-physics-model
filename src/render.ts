// Canvas drawing: sea and coast, oil by the Bonn Agreement appearance code, drift tracers, and the things a person
// has placed (booms, wind and current paths, leaks). The grid is north-up; y grows upwards in metres.
import { DX, HALF, N, cellCentre, type Pt, type Scene } from './world';

/** Bonn Agreement classes, lower bound in metres of thickness, with the colour drawn. */
export const BONN: { name: string; min: number; css: string }[] = [
  { name: 'Sheen', min: 0.04e-6, css: '#c9d1d9' },
  { name: 'Rainbow', min: 0.3e-6, css: 'linear-gradient(90deg,#c084fc,#60a5fa,#4ade80,#facc15)' },
  { name: 'Metallic', min: 5e-6, css: 'rgb(130,134,140)' },
  { name: 'Broken true colour', min: 50e-6, css: 'rgb(110,66,32)' },
  { name: 'Continuous true colour', min: 200e-6, css: 'rgb(42,24,12)' },
];

/** Colour lookup over log10 thickness from −7.5 (0.03 µm) to −2.5 (3 mm), RGBA packed little-endian. */
const LUT = (() => {
  const lut = new Uint32Array(256);
  const hsl = (h: number, s: number, l: number): [number, number, number] => {
    const a = s * Math.min(l, 1 - l), f = (n: number) => { const k = (n + h / 30) % 12; return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
    return [f(0) * 255, f(8) * 255, f(4) * 255];
  };
  for (let b = 0; b < 256; b++) {
    const h = 10 ** (-7.5 + (5 * b) / 255);
    let c: [number, number, number, number];
    if (h < 0.04e-6) c = [0, 0, 0, 0];
    else if (h < 0.3e-6) c = [205, 215, 225, 60 + 40 * Math.log10(h / 0.04e-6)];
    else if (h < 5e-6) { const t = Math.log10(h / 0.3e-6) / Math.log10(5 / 0.3); c = [...hsl(290 - 250 * t, 0.75, 0.62), 130]; }
    else if (h < 50e-6) { const t = Math.log10(h / 5e-6); c = [150 - 40 * t, 154 - 44 * t, 160 - 42 * t, 185]; }
    else if (h < 200e-6) c = [110, 66, 32, 220];
    else c = [42, 24, 12, 245];
    lut[b] = (Math.round(c[3]) << 24) | (Math.round(c[2]) << 16) | (Math.round(c[1]) << 8) | Math.round(c[0]);
  }
  return lut;
})();

const rand = (i: number) => { const s = Math.sin(i * 12.9898) * 43758.5453; return s - Math.floor(s); };

export interface Overlay {
  scene: Scene;
  selected: { kind: 'boom' | 'stroke' | 'source'; id: number } | null;
  draft: { tool: string; path: Pt[]; radius?: number } | null;
  time: number;
  /** Optional arrow fields on the 16 × 16 node grid, [u, v] m/s per node. */
  arrows?: { wind: Float32Array | null; current: Float32Array | null };
}

export class View {
  readonly ctx: CanvasRenderingContext2D;
  S = 1; ox = 0; oy = 0; dpr = 1;
  private base = document.createElement('canvas');
  private oil = document.createElement('canvas');
  private oilCtx: CanvasRenderingContext2D;
  private oilImg: ImageData;
  private trail = document.createElement('canvas');
  private trailCtx: CanvasRenderingContext2D;
  private tracers = new Float64Array(1400 * 3); // x, y, life (s)
  private land: Uint8Array = new Uint8Array(N * N);

  constructor(readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    this.base.width = this.base.height = this.oil.width = this.oil.height = N;
    this.oilCtx = this.oil.getContext('2d')!;
    this.oilImg = this.oilCtx.createImageData(N, N);
    this.trailCtx = this.trail.getContext('2d')!;
    for (let p = 0; p < this.tracers.length; p += 3) this.respawn(p, rand(p) * 6);
  }

  resize() {
    const r = this.canvas.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
    if (this.canvas.width !== Math.round(r.width * dpr) || this.canvas.height !== Math.round(r.height * dpr)) {
      this.canvas.width = Math.round(r.width * dpr);
      this.canvas.height = Math.round(r.height * dpr);
      this.trail.width = this.canvas.width;
      this.trail.height = this.canvas.height;
    }
    this.dpr = dpr;
    this.S = Math.max(50, Math.min(r.width, r.height) - 28);
    this.ox = (r.width - this.S) / 2;
    this.oy = (r.height - this.S) / 2;
  }

  /** Metres → CSS px. */
  px([x, y]: Pt): Pt { return [this.ox + ((x + HALF) / (2 * HALF)) * this.S, this.oy + ((HALF - y) / (2 * HALF)) * this.S]; }
  /** CSS px → metres. */
  m(px: number, py: number): Pt { return [((px - this.ox) / this.S) * 2 * HALF - HALF, HALF - ((py - this.oy) / this.S) * 2 * HALF]; }
  /** Metres → CSS px, as a length. */
  len(metres: number) { return (metres / (2 * HALF)) * this.S; }

  setLand(land: Uint8Array) {
    this.land = land;
    const c = this.base.getContext('2d')!, img = c.createImageData(N, N), d = new Uint32Array(img.data.buffer);
    const at = (i: number, j: number) => land[Math.min(N - 1, Math.max(0, j)) * N + Math.min(N - 1, Math.max(0, i))];
    for (let j = 0; j < N; j++)
      for (let i = 0; i < N; i++) {
        const k = j * N + i, row = N - 1 - j, n = rand(k) * 10;
        let r: number, g: number, b: number;
        if (land[k]) {
          let near = 9;
          for (let d2 = 1; d2 <= 3 && near === 9; d2++) if (!at(i + d2, j) || !at(i - d2, j) || !at(i, j + d2) || !at(i, j - d2)) near = d2;
          [r, g, b] = near <= 1 ? [196, 176, 128] : near <= 3 ? [128, 132, 92] : [78, 96, 66];
          r += n; g += n; b += n * 0.6;
        } else {
          const shore = at(i + 1, j) || at(i - 1, j) || at(i, j + 1) || at(i, j - 1) || at(i + 2, j) || at(i - 2, j) || at(i, j + 2) || at(i, j - 2);
          [r, g, b] = shore ? [22, 78, 96] : [9, 38, 58];
          b += n * 0.8; g += n * 0.4;
        }
        d[row * N + i] = (255 << 24) | (b << 16) | (g << 8) | r;
      }
    c.putImageData(img, 0, 0);
  }

  private respawn(p: number, life = 2 + rand(p + performance.now()) * 4) {
    for (let tries = 0; tries < 20; tries++) {
      const x = (Math.random() * 2 - 1) * HALF, y = (Math.random() * 2 - 1) * HALF;
      const i = Math.floor((x + HALF) / DX), j = Math.floor((y + HALF) / DX);
      if (!this.land[j * N + i]) { this.tracers[p] = x; this.tracers[p + 1] = y; break; }
    }
    this.tracers[p + 2] = life;
  }

  /** Bilinear drift at (x, y) from the 16 × 16 node grid (nodes at cell 8, 24, ...). */
  private driftAt(drift: Float32Array, x: number, y: number): Pt {
    const gx = Math.min(14.999, Math.max(0, ((x + HALF) / DX - 8.5) / 16)), gy = Math.min(14.999, Math.max(0, ((y + HALF) / DX - 8.5) / 16));
    const i = Math.floor(gx), j = Math.floor(gy), fx = gx - i, fy = gy - j;
    const at = (a: number, b: number, c: number) => drift[2 * (b * 16 + a) + c];
    const f = (c: number) => (1 - fy) * ((1 - fx) * at(i, j, c) + fx * at(i + 1, j, c)) + fy * ((1 - fx) * at(i, j + 1, c) + fx * at(i + 1, j + 1, c));
    return [f(0), f(1)];
  }

  draw(h: Float32Array | null, drift: Float32Array | null, simRate: number, dt: number, o: Overlay) {
    const { ctx, dpr, S, ox, oy } = this, W = this.canvas.width / dpr, H = this.canvas.height / dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.save();
    ctx.shadowColor = 'rgba(16,24,32,0.22)'; ctx.shadowBlur = 18; ctx.shadowOffsetY = 4;
    ctx.drawImage(this.base, ox, oy, S, S);
    ctx.restore();

    // Drift tracers: faint trails that show where the surface is going.
    const tc = this.trailCtx;
    tc.setTransform(dpr, 0, 0, dpr, 0, 0);
    tc.globalCompositeOperation = 'destination-out';
    tc.fillStyle = `rgba(0,0,0,${Math.min(1, dt * 2.2)})`;
    tc.fillRect(0, 0, W, H);
    tc.globalCompositeOperation = 'source-over';
    if (drift) {
      tc.strokeStyle = 'rgba(190,228,255,0.5)';
      tc.lineWidth = 1;
      tc.beginPath();
      const T = this.tracers;
      for (let p = 0; p < T.length; p += 3) {
        T[p + 2] -= dt;
        const [u, v] = this.driftAt(drift, T[p], T[p + 1]);
        const x2 = T[p] + u * simRate * dt, y2 = T[p + 1] + v * simRate * dt;
        const i = Math.floor((x2 + HALF) / DX), j = Math.floor((y2 + HALF) / DX);
        if (T[p + 2] <= 0 || i < 0 || j < 0 || i >= N || j >= N || this.land[j * N + i]) { this.respawn(p); continue; }
        const [ax, ay] = this.px([T[p], T[p + 1]]), [bx, by] = this.px([x2, y2]);
        tc.moveTo(ax, ay); tc.lineTo(bx + 0.01, by);
        T[p] = x2; T[p + 1] = y2;
      }
      tc.stroke();
    }
    ctx.drawImage(this.trail, 0, 0, W, H);

    if (h) {
      const d = new Uint32Array(this.oilImg.data.buffer);
      for (let j = 0; j < N; j++) {
        const row = (N - 1 - j) * N;
        for (let i = 0; i < N; i++) {
          const v = h[j * N + i];
          d[row + i] = v < 0.03e-6 ? 0 : LUT[Math.max(0, Math.min(255, Math.round(((Math.log10(v) + 7.5) * 255) / 5)))];
        }
      }
      this.oilCtx.putImageData(this.oilImg, 0, 0);
      ctx.drawImage(this.oil, ox, oy, S, S);
    }

    ctx.strokeStyle = '#ffffff14';
    ctx.lineWidth = 1;
    ctx.strokeRect(ox + 0.5, oy + 0.5, S - 1, S - 1);
    if (o.arrows?.current) this.field(o.arrows.current, '#5ef0b0', 120, 0.01);
    if (o.arrows?.wind) this.field(o.arrows.wind, '#e6f4ff', 1.6, 0.3);
    this.overlays(o);
  }

  /** Arrows on the 16 × 16 node grid: px per m/s, capped at 30 px; nodes on land or below `min` m/s are skipped. */
  private field(f: Float32Array, color: string, pxPerMs: number, min: number) {
    const c = this.ctx, k = this.S / 800;
    c.strokeStyle = color; c.fillStyle = color; c.lineWidth = 1.6; c.lineCap = 'round';
    for (let g = 0; g < 256; g++) {
      const i = 8 + 16 * (g % 16), j = 8 + 16 * Math.floor(g / 16);
      if (this.land[j * N + i]) continue;
      const u = f[2 * g], v = f[2 * g + 1], sp = Math.hypot(u, v);
      if (sp < min) continue;
      const L = Math.min(30, sp * pxPerMs) * k, [x, y] = this.px([cellCentre(i), cellCentre(j)]);
      const dx = (u / sp) * L, dy = -(v / sp) * L;
      c.beginPath(); c.moveTo(x - dx / 2, y - dy / 2); c.lineTo(x + dx / 2, y + dy / 2); c.stroke();
      const a = Math.atan2(dy, dx);
      c.save(); c.translate(x + dx / 2, y + dy / 2); c.rotate(a);
      c.beginPath(); c.moveTo(1, 0); c.lineTo(-5, -3.5); c.lineTo(-5, 3.5); c.closePath(); c.fill();
      c.restore();
    }
  }

  private path(pts: Pt[]) {
    const c = this.ctx;
    c.beginPath();
    pts.forEach((p, i) => { const [x, y] = this.px(p); if (i) c.lineTo(x, y); else c.moveTo(x, y); });
  }

  /** Arrowheads every ~110 px along a path, pointing the way it was drawn. */
  private heads(pts: Pt[], color: string) {
    const c = this.ctx, px = pts.map((p) => this.px(p));
    let carry = 55;
    c.fillStyle = color;
    for (let s = 1; s < px.length; s++) {
      const [ax, ay] = px[s - 1], [bx, by] = px[s], L = Math.hypot(bx - ax, by - ay);
      let d = carry;
      while (d < L) {
        const x = ax + ((bx - ax) * d) / L, y = ay + ((by - ay) * d) / L, a = Math.atan2(by - ay, bx - ax);
        c.save(); c.translate(x, y); c.rotate(a);
        c.beginPath(); c.moveTo(7, 0); c.lineTo(-5, -5); c.lineTo(-2, 0); c.lineTo(-5, 5); c.closePath(); c.fill();
        c.restore();
        d += 110;
      }
      carry = d - L;
    }
  }

  private overlays(o: Overlay) {
    const c = this.ctx, sel = o.selected;
    for (const s of o.scene.strokes) {
      const color = s.kind === 'wind' ? '#7dd3fc' : '#6ee7b7', on = sel?.kind === 'stroke' && sel.id === s.id;
      c.lineJoin = 'round'; c.lineCap = 'round';
      this.path(s.path);
      c.strokeStyle = s.kind === 'wind' ? 'rgba(125,211,252,0.07)' : 'rgba(110,231,183,0.08)';
      c.lineWidth = Math.max(4, this.len(s.width));
      c.stroke();
      c.setLineDash([10, 8]);
      c.lineDashOffset = -o.time * (s.kind === 'wind' ? 40 : 18);
      c.strokeStyle = color;
      c.globalAlpha = on ? 1 : 0.7;
      c.lineWidth = on ? 2.5 : 1.6;
      c.stroke();
      c.setLineDash([]);
      this.heads(s.path, color);
      c.globalAlpha = 1;
    }
    for (const b of o.scene.booms) {
      const on = sel?.kind === 'boom' && sel.id === b.id;
      c.lineJoin = 'round'; c.lineCap = 'round';
      this.path(b.path);
      c.strokeStyle = on ? '#ffffff' : '#1c1403';
      c.lineWidth = on ? 9 : 7;
      c.stroke();
      c.strokeStyle = '#fbbf24';
      c.lineWidth = 4;
      c.stroke();
      c.setLineDash([2, 10]);
      c.strokeStyle = '#7c2d12';
      c.lineWidth = 4;
      c.stroke();
      c.setLineDash([]);
    }
    for (const s of o.scene.sources) {
      const [x, y] = this.px(s.at), on = sel?.kind === 'source' && sel.id === s.id;
      for (let k = 0; k < 2; k++) {
        const ph = (o.time * 0.6 + k / 2) % 1;
        c.beginPath(); c.arc(x, y, 6 + ph * 22, 0, 2 * Math.PI);
        c.strokeStyle = `rgba(248,113,113,${0.8 * (1 - ph)})`; c.lineWidth = 2; c.stroke();
      }
      c.beginPath(); c.arc(x, y, 6, 0, 2 * Math.PI);
      c.fillStyle = '#f87171'; c.fill();
      c.strokeStyle = on ? '#fff' : '#450a0a'; c.lineWidth = 2; c.stroke();
    }
    const d = o.draft;
    if (d && d.path.length) {
      c.setLineDash([6, 6]);
      c.strokeStyle = d.tool === 'boom' ? '#fbbf24' : d.tool === 'current' || d.tool === 'eddy' ? '#6ee7b7' : d.tool === 'spill' ? '#f59e0b' : '#7dd3fc';
      c.lineWidth = 2;
      if (d.radius !== undefined) {
        const [x, y] = this.px(d.path[0]);
        c.beginPath(); c.arc(x, y, this.len(d.radius), 0, 2 * Math.PI); c.stroke();
      } else { this.path(d.path); c.stroke(); }
      c.setLineDash([]);
    }
  }
}
