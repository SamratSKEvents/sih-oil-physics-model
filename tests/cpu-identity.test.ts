// CPU engine speed-ups must not change a single bit: every scenario, with live edits, against recorded fingerprints.
import { expect, test } from 'vitest';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { replay, type LogEntry } from '../src/runner';
import { SCENARIOS, spiral } from '../src/world';

const FILE = new URL('./cpu-fingerprints.json', import.meta.url);

function run(id: string) {
  const sc = SCENARIOS.find((s) => s.id === id)!;
  const scene = structuredClone(sc.scene);
  scene.booms.push({ id: 90, path: [[-500, -2500], [300, -1500]] });
  scene.strokes.push({ id: 91, kind: 'wind', label: 's', width: 1200, speed: 12, path: spiral([1500, 1500], 2000) });
  const log: LogEntry[] = [
    { chunk: 10, edit: { k: 'param', key: 'windDirDeg', value: 150 } },
    { chunk: 20, edit: { k: 'scene', scene } },
    { chunk: 25, edit: { k: 'spill', spill: { at: [1000, -1000], radius: 400, volume: 150 } } },
    { chunk: 30, edit: { k: 'param', key: 'eddySpeed', value: 0.07 } },
  ];
  const out: string[] = [];
  replay(sc, log, 60, (r) => out.push(r.fingerprint()));
  return out;
}

test('CPU engine output is unchanged', () => {
  // scenarios on record must match; a scenario added since is recorded (the engine is what is under test)
  const saved: Record<string, string[]> = existsSync(FILE) ? JSON.parse(readFileSync(FILE, 'utf8')) : {};
  for (const s of SCENARIOS) {
    const now = run(s.id);
    if (saved[s.id]) expect(now, s.id).toEqual(saved[s.id]);
    else saved[s.id] = now;
  }
  writeFileSync(FILE, JSON.stringify(saved));
}, 120000);
