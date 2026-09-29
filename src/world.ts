// The sandbox: a 12.8 km square of sea on the engine's 256 × 256 grid of 50 m cells, the scenarios that set it up,
// and the edits a person can make while it runs. Shared by the page and the simulation worker, so both build the same
// coast from the same scenario id. Coordinates are metres, x east and y north of the centre.

export const N = 256;
export const DX = 50;
export const HALF = (N * DX) / 2;
/** Simulated seconds per step. Every edit lands on a step boundary, which is what makes a run replayable. */
export const CHUNK = 60;

export type Pt = [number, number];

export interface Boom { id: number; path: Pt[] }
export interface Stroke { id: number; kind: 'wind' | 'current'; path: Pt[]; width: number; speed: number; label: string }
export interface Source { id: number; at: Pt; rate: number } // m³/h
export interface Scene { booms: Boom[]; strokes: Stroke[]; sources: Source[] }
export interface Spill { at: Pt; radius: number; volume: number }

export type Edit =
  | { k: 'scene'; scene: Scene }
  | { k: 'param'; key: string; value: number }
  | { k: 'spill'; spill: Spill };

export interface Scenario {
  id: string;
  name: string;
  blurb: string;
  params: Record<string, number>;
  land?: (x: number, y: number) => boolean;
  scene: Scene;
  spills: Spill[];
}

export const cellCentre = (i: number) => (i + 0.5 - N / 2) * DX;

export function landMask(sc: Scenario): Uint8Array {
  const m = new Uint8Array(N * N);
  if (sc.land) for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) m[j * N + i] = sc.land(cellCentre(i), cellCentre(j)) ? 1 : 0;
  return m;
}

/** An inward spiral from radius R to 0.15 R, anticlockwise: the inflow of a northern-hemisphere cyclone. */
export function spiral([cx, cy]: Pt, R: number, turns = 2.5): Pt[] {
  const pts: Pt[] = [], n = Math.round(turns * 48);
  for (let s = 0; s <= n; s++) {
    const a = (s / n) * turns * 2 * Math.PI, r = R * (1 - 0.85 * (s / n));
    pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return pts;
}

/** A closed anticlockwise circle: an eddy when used as a current. */
export function ring([cx, cy]: Pt, R: number): Pt[] {
  const pts: Pt[] = [];
  for (let s = 0; s <= 64; s++) { const a = (s / 64) * 2 * Math.PI; pts.push([cx + R * Math.cos(a), cy + R * Math.sin(a)]); }
  return pts;
}

/** Dashes of boom along an ellipse: n pieces, each `arc` degrees long, the first starting at `from` degrees. */
function dashedRing(firstId: number, rx: number, ry: number, n: number, arc: number, from: number): Scene['booms'] {
  return Array.from({ length: n }, (_, d) => {
    const a0 = ((from + (d * 360) / n) * Math.PI) / 180, path: Pt[] = [];
    for (let s = 0; s <= 8; s++) { const a = a0 + ((arc * Math.PI) / 180) * (s / 8); path.push([rx * Math.cos(a), ry * Math.sin(a)]); }
    return { id: firstId + d, path };
  });
}

const empty = (): Scene => ({ booms: [], strokes: [], sources: [] });

export const SCENARIOS: Scenario[] = [
  {
    id: 'boom-rings',
    name: 'Boom rings',
    blurb: 'A slick inside two broken rings of boom, gaps staggered. Wind and eddies push oil through the inner gaps; the outer ring catches most of it.',
    params: { windSpeed: 7, windDirDeg: 60, eddySpeed: 0.05, uTide: 0.1 },
    scene: {
      booms: [...dashedRing(1, 2600, 1750, 6, 36, 15), ...dashedRing(20, 5000, 3300, 8, 26, 38)],
      strokes: [],
      sources: [],
    },
    spills: [{ at: [0, 0], radius: 750, volume: 400 }],
  },
  {
    id: 'four-way',
    name: 'Four-way split',
    blurb: 'Four winds blow out from the middle, to each corner. The slick is pulled four ways at once and tears into four.',
    params: { windSpeed: 0, windDirDeg: 0, eddySpeed: 0.02, uTide: 0 },
    scene: {
      booms: [], sources: [],
      strokes: ([[-1, 1, 'north-west'], [1, 1, 'north-east'], [-1, -1, 'south-west'], [1, -1, 'south-east']] as const).map(([sx, sy, name], i) => ({
        // each path starts ~850 m out and is narrow enough to leave the centre calm: where paths overlap the last
        // drawn wins, so overlapping at the middle would hand the whole slick to one corner
        id: 50 + i, kind: 'wind' as const, label: `Wind to the ${name}`, width: 900, speed: 20,
        path: [[sx * 600, sy * 600], [sx * 5200, sy * 5200]] as Pt[],
      })),
    },
    spills: [{ at: [0, 0], radius: 1100, volume: 450 }],
  },
  {
    id: 'open-sea',
    name: 'Open sea',
    blurb: 'A fresh release in open water. Watch it stretch downwind, streak along windrows and tear into patches.',
    params: { windSpeed: 8, windDirDeg: 60, eddySpeed: 0.04, uTide: 0.1 },
    scene: empty(),
    spills: [{ at: [-2500, -2200], radius: 700, volume: 400 }],
  },
  {
    id: 'canal',
    name: 'Canal intake',
    blurb: 'A current pulls water into a canal. The slick is drawn to the mouth, squeezed through and carried round the bend.',
    params: { windSpeed: 6, windDirDeg: 90, eddySpeed: 0.03, uTide: 0.3 },
    land: (x, y) => x > -1200 && !(Math.abs(y) < 230 && x < 4420) && !(x > 3980 && x < 4420 && y > 0 && y < 5400),
    scene: {
      booms: [], sources: [], strokes: [
        { id: 1, kind: 'current', label: 'Intake draw', width: 1600, speed: 0.3, path: [[-4200, -1600], [-1500, 0]] },
        { id: 2, kind: 'current', label: 'Canal flow', width: 520, speed: 0.45, path: [[-1600, 0], [4200, 0], [4200, 5200]] },
      ],
    },
    spills: [{ at: [-3700, -1400], radius: 600, volume: 350 }],
  },
  {
    id: 'harbour',
    name: 'Harbour boom',
    blurb: 'Onshore wind pushes the slick at a harbour mouth. A boom holds it off. Select the boom and press Delete to see why it is there.',
    params: { windSpeed: 9, windDirDeg: 80, eddySpeed: 0.03, uTide: 0.15 },
    land: (x, y) => {
      if (x < 2200 + 300 * Math.sin(y / 1800)) return false;
      const basin = x > 3000 && x < 5400 && y > -2200 && y < 1600;
      const mouth = x < 3100 && Math.abs(y + 300) < 360;
      return !basin && !mouth;
    },
    scene: { booms: [{ id: 1, path: [[1850, -1500], [1450, -700], [1400, 100], [1850, 900]] }], strokes: [], sources: [] },
    spills: [{ at: [-3000, -800], radius: 700, volume: 450 }],
  },
  {
    id: 'cyclone',
    name: 'Spiral wind',
    blurb: 'A cyclonic wind spirals in towards a centre. Two slicks get wound round it and stretched into arms.',
    params: { windSpeed: 4, windDirDeg: 30, eddySpeed: 0.03, uTide: 0.05, windage: 0.035 },
    scene: { booms: [], sources: [], strokes: [{ id: 1, kind: 'wind', label: 'Spiral wind', width: 1500, speed: 16, path: spiral([0, 0], 5600) }] },
    spills: [{ at: [-1800, 2600], radius: 600, volume: 300 }, { at: [2200, -2000], radius: 600, volume: 300 }],
  },
  {
    id: 'island',
    name: 'Island wake',
    blurb: 'A steady current meets an island. The slick splits round it, strands on the upstream shore and rejoins in the wake.',
    params: { windSpeed: 3, windDirDeg: 90, eddySpeed: 0.05, uTide: 0.05 },
    land: (x, y) => (x + 500) ** 2 + y ** 2 < 1300 ** 2 || (x - 2300) ** 2 + (y - 1700) ** 2 < 450 ** 2,
    scene: { booms: [], sources: [], strokes: [{ id: 1, kind: 'current', label: 'Through-current', width: 7000, speed: 0.3, path: [[-6300, -200], [6300, -200]] }] },
    spills: [{ at: [-4300, -300], radius: 750, volume: 500 }],
  },
  {
    id: 'strait',
    name: 'Leak in a strait',
    blurb: 'A seabed leak keeps feeding oil into a narrow strait. The tide sloshes it back and forth through the neck.',
    params: { windSpeed: 5, windDirDeg: 0, eddySpeed: 0.03, uTide: 0.5 },
    land: (x, y) => Math.abs(x) > 700 + 3200 * (y / HALF) ** 2,
    scene: { booms: [], strokes: [], sources: [{ id: 1, at: [0, -4200], rate: 60 }] },
    spills: [],
  },
];
