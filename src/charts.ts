// Live time-series charts for the side panel, plain SVG. One chart per measure (never two y-scales): the oil budget
// as a stacked area, slick area and thickest oil as single lines. Each has a hover crosshair and readout; the
// budget's legend carries the current values, so no series is told apart by colour alone.

export interface Series { key: string; label: string; color: string }

interface Opts {
  series: Series[];
  stacked?: boolean;
  unit: string;
  fmt: (v: number) => string;
  /** Label for an x value (simulated seconds). */
  xfmt: (x: number) => string;
}

const W = 248, H = 96, PAD = { l: 34, r: 6, t: 6, b: 16 };
const NS = 'http://www.w3.org/2000/svg';

/** A round top for the y-axis: 1, 2 or 5 × 10^n at or above v. */
function niceMax(v: number) {
  if (!(v > 0)) return 1;
  const e = 10 ** Math.floor(Math.log10(v)), m = v / e;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * e;
}

export class TimeChart {
  private svg: SVGSVGElement;
  private tip: HTMLDivElement;
  private xs: number[] = [];
  private ys: number[][] = [];
  private hover: number | null = null;

  constructor(host: HTMLElement, private o: Opts) {
    host.classList.add('chart');
    this.svg = document.createElementNS(NS, 'svg');
    this.svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    this.svg.setAttribute('role', 'img');
    this.tip = document.createElement('div');
    this.tip.className = 'chart-tip';
    this.tip.hidden = true;
    host.append(this.svg, this.tip);
    const at = (e: PointerEvent) => {
      if (this.xs.length < 2) return null;
      const r = this.svg.getBoundingClientRect(), x = ((e.clientX - r.left) / r.width) * W;
      const f = Math.max(0, Math.min(1, (x - PAD.l) / (W - PAD.l - PAD.r)));
      return Math.round(f * (this.xs.length - 1));
    };
    this.svg.addEventListener('pointermove', (e) => { this.hover = at(e); this.draw(); });
    this.svg.addEventListener('pointerleave', () => { this.hover = null; this.draw(); });
  }

  /** Replace the data: x in simulated seconds, one y array per series. */
  set(xs: number[], ys: number[][]) {
    this.xs = xs;
    this.ys = ys;
    this.draw();
  }

  private draw() {
    const { xs, ys, o } = this, n = xs.length;
    const x0 = xs[0] ?? 0, x1 = Math.max(xs[n - 1] ?? 1, x0 + 1);
    const tops = xs.map((_, i) => (o.stacked ? ys.reduce((s, y) => s + Math.max(0, y[i]), 0) : Math.max(...ys.map((y) => y[i]))));
    const yMax = niceMax(Math.max(0, ...tops));
    const X = (x: number) => PAD.l + ((x - x0) / (x1 - x0)) * (W - PAD.l - PAD.r);
    const Y = (y: number) => H - PAD.b - (y / yMax) * (H - PAD.t - PAD.b);
    let out = '';
    // recessive grid: baseline, middle, top
    for (const g of [0, 0.5, 1]) {
      const y = Y(g * yMax);
      out += `<line x1="${PAD.l}" x2="${W - PAD.r}" y1="${y}" y2="${y}" class="grid"/>`;
      out += `<text x="${PAD.l - 4}" y="${y + 3}" class="axis" text-anchor="end">${o.fmt(g * yMax)}</text>`;
    }
    out += `<text x="${PAD.l}" y="${H - 3}" class="axis">${o.xfmt(x0)}</text><text x="${W - PAD.r}" y="${H - 3}" class="axis" text-anchor="end">${o.xfmt(x1)}</text>`;
    if (n >= 2) {
      if (o.stacked) {
        const base = new Array(n).fill(0);
        ys.forEach((y, s) => {
          const lo = base.slice(), hi = base.map((b, i) => b + Math.max(0, y[i]));
          const top = xs.map((x, i) => `${X(x).toFixed(1)},${Y(hi[i]).toFixed(1)}`), bot = xs.map((x, i) => `${X(x).toFixed(1)},${Y(lo[i]).toFixed(1)}`).reverse();
          // a 2 px surface gap between layers keeps neighbouring fills apart
          out += `<polygon points="${top.join(' ')} ${bot.join(' ')}" fill="${o.series[s].color}" stroke="var(--surface)" stroke-width="1" stroke-linejoin="round"/>`;
          for (let i = 0; i < n; i++) base[i] = hi[i];
        });
      } else {
        ys.forEach((y, s) => {
          out += `<polyline points="${xs.map((x, i) => `${X(x).toFixed(1)},${Y(y[i]).toFixed(1)}`).join(' ')}" fill="none" stroke="${o.series[s].color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
        });
      }
    }
    const h = this.hover;
    if (h !== null && n >= 2) {
      const x = X(xs[h]);
      out += `<line x1="${x}" x2="${x}" y1="${PAD.t}" y2="${H - PAD.b}" class="cross"/>`;
      if (!o.stacked) ys.forEach((y, s) => { out += `<circle cx="${x}" cy="${Y(y[h])}" r="4" fill="${o.series[s].color}" stroke="var(--surface)" stroke-width="2"/>`; });
      this.tip.hidden = false;
      this.tip.innerHTML = `<b>${o.xfmt(xs[h])}</b>` + o.series.map((s, i) => `<div><i style="background:${s.color}"></i>${s.label}<span>${o.fmt(ys[i][h])} ${o.unit}</span></div>`).join('');
      this.tip.style.left = `${Math.min(62, Math.max(4, (x / W) * 100 - 18))}%`;
    } else this.tip.hidden = true;
    this.svg.innerHTML = out;
  }
}
