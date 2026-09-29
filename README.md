# Slick Lab

A live oil-film simulator on a synthetic 12.8 km sea (256 × 256 cells of 50 m). No map, no data: a canvas, six
scenarios and tools to change the sea while it runs.

```bash
npm install
npm run dev     # http://localhost:5288
npm test        # replays a run with live edits and checks it is bit-identical
npm run build
```

or double-click `run.bat`.

## What you can do

- **Scenarios:** open sea, a canal intake, a harbour boom, a spiral wind, an island wake, a leak in a tidal strait.
- **Tools:** lay booms, draw wind or current paths (they follow your stroke), drop a spiral wind or a current eddy,
  spill oil or place a leak that keeps releasing. Everything placed can be dragged, tuned or deleted while it runs.
- **Conditions:** wind speed and direction, background and tidal current, eddies, windage, diffusion, film break-up.
- **Timeline:** scrub back through the recording, then resume from any point.
- **Determinism:** *Verify this run* replays the whole edit log from zero in a second worker and compares a
  fingerprint of every cell, step by step.

## How it works

`src/engine/` is the slick model, copied unchanged from the oil engine (`packages/engine/src`, only the files the
model imports). It solves an oil layer with its own thickness and momentum over a tidal water layer, with film
rupture, turbulent diffusion, windage, windrows, eddies, natural dispersion, evaporation and stranding.

`src/runner.ts` puts that model on a scenario's coast and applies edits. The simulation advances in fixed
one-minute steps, and every edit is logged against the step it landed on, so a scenario plus its log is the run.
`src/worker.ts` runs it off the main thread; `src/main.ts` and `src/render.ts` are the page.

Everything here is synthetic: coasts, currents and coefficients are assumptions, not a forecast.
