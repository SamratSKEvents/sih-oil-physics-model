// The claim the app makes: a scenario plus its edit log reproduces the run bit for bit.
import { expect, test } from 'vitest';
import { Runner, replay, type LogEntry } from './runner';
import { SCENARIOS } from './world';

test('a run with live edits replays bit-identically', () => {
  const sc = SCENARIOS.find((s) => s.id === 'harbour')!;
  const log: LogEntry[] = [
    { chunk: 3, edit: { k: 'param', key: 'windDirDeg', value: 120 } },
    { chunk: 5, edit: { k: 'scene', scene: { ...sc.scene, booms: [] } } },
    { chunk: 5, edit: { k: 'param', key: 'eddySpeed', value: 0.08 } },
    { chunk: 8, edit: { k: 'spill', spill: { at: [0, 0], radius: 400, volume: 100 } } },
  ];
  const live = new Runner(sc), seen: string[] = [];
  let e = 0;
  for (let c = 0; c < 12; c++) {
    while (e < log.length && log[e].chunk === live.chunk) live.apply(log[e++].edit);
    live.step();
    seen.push(live.fingerprint());
  }
  const again: string[] = [];
  replay(sc, log, 12, (r) => again.push(r.fingerprint()));
  expect(again).toEqual(seen);
  expect(new Set(seen).size).toBe(12); // the oil actually moved
});
