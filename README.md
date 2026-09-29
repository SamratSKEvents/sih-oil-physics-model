# Slick Lab

A live oil-film simulator on a synthetic 12.8 km sea (256 × 256 cells of 50 m). No map, no data: a canvas, six
scenarios and tools to change the sea while it runs.

```bash
npm install
npm run dev     # http://localhost:5288
npm test        # determinism, and that the CPU engine's output is unchanged bit for bit
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
- **Two engines:** WebGPU compute shaders by default, the CPU engine where WebGPU is missing or when picked in the
  header. Speeds go to 6000× and *Max*; the clock shows the rate actually reached.

## How it works

`src/engine/` is the slick model, copied unchanged from the oil engine (`packages/engine/src`, only the files the
model imports). It solves an oil layer with its own thickness and momentum over a tidal water layer, with film
rupture, turbulent diffusion, windage, windrows, eddies, natural dispersion, evaporation and stranding.

`src/runner.ts` puts that model on a scenario's coast and applies edits. The simulation advances in fixed
one-minute steps, and every edit is logged against the step it landed on, so a scenario plus its log is the run.
`src/worker.ts` runs it off the main thread; `src/main.ts` and `src/render.ts` are the page.

Everything here is synthetic: coasts, currents and coefficients are assumptions, not a forecast.

## GPU engine

`src/gpu/` is the same step as WebGPU compute kernels (`kernels.ts`, a line-for-line port of the CPU code it names),
driven by `sim.ts`. It runs in 32-bit floats; the CPU engine is 64-bit. Each simulated minute is one submission, and
minute c sizes its substeps from the fastest oil speed read back at the end of minute c − 2, so two minutes are in
flight and readback latency is hidden. That rule depends only on step numbers, and nothing uses atomics or
order-dependent sums, so a GPU run replays bit-identically on the same GPU and browser. A run recorded on one GPU is
not expected to replay bit-for-bit on another.

Where the port cannot be literal it says so in the kernel: stranding is resolved in three passes (ask, grant, give)
instead of the CPU's cell-by-cell walk, and windrow lines use an integer hash in place of the CPU's `sin` hash.

`/tools/validate.html` (with `npm run dev`) runs every scenario on both engines and compares them. On a Radeon RDNA2
laptop GPU, after 4 simulated hours: 7–12× faster per simulated minute (5–8 ms against 40–90 ms), floating volume and
evaporation within 0.1 %, sheen area within 5 %, footprints overlapping 0.78–0.98. The narrow canal differs most
(floating +6 %, stranded −17 %): in a 9-cell-wide channel small differences in where the oil meets the wall grow.

