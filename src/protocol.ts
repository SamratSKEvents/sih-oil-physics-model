// Messages between the page and the simulation worker.
import type { LogEntry } from './runner';
import type { Edit, Scene } from './world';

/** Where the model runs: WebGPU compute, or the CPU engine as written. */
export type Engine = 'gpu' | 'cpu';

export type ToWorker =
  | { type: 'load'; scenario: string; gen: number; engine: Engine }
  | { type: 'edit'; edit: Edit }
  | { type: 'play'; playing: boolean; speed: number }
  | { type: 'step' }
  | { type: 'seek'; chunk: number; gen: number }
  | { type: 'export' }
  | { type: 'verify'; scenario: string; engine: Engine; log: LogEntry[]; upTo: number; hashes: [number, string][] };

export interface Budget { released: number; floating: number; evaporated: number; dispersed: number; stranded: number; left: number; held: number }

export interface FrameMsg {
  type: 'frame';
  gen: number;
  engine: Engine;
  chunk: number;
  t: number;
  playing: boolean;
  /** Oil thickness, m, row-major from the south-west. */
  h: Float32Array;
  /** Surface drift, 16 × 16 × [u, v] m/s. */
  drift: Float32Array;
  /** Fingerprints of the steps run since the last frame. */
  hashes: [number, string][];
  budget: Budget;
  marks?: { chunk: number; k: Edit['k'] }[];
  stats?: [string, string][];
  warnings?: string[];
}

export type FromWorker =
  | FrameMsg
  | { type: 'scene'; scene: Scene; params: Record<string, number>; gen: number; engine: Engine }
  | { type: 'progress'; done: number; of: number; gen?: number }
  | { type: 'log'; scenario: string; engine: Engine; log: LogEntry[]; chunk: number }
  | { type: 'verified'; checked: number; first: number; engine: Engine };
