export interface EnvironmentSample {
  currentU: number; // m/s east
  currentV: number; // m/s north
  windU: number; // m/s east, 10 m
  windV: number; // m/s north, 10 m
  stokesU: number; // m/s east
  stokesV: number; // m/s north
  waveHeight: number; // m
  sst: number; // °C
  landMask: number; // 0 water .. 1 land (bilinear blend of the 0/1 node mask)
  outOfDomain: boolean;
}

export interface EnvironmentProvider {
  /** Deterministic pure function of (lat, lon, t). `out` is an optional object to fill (avoids allocation). */
  sample(lat: number, lon: number, timeSeconds: number, out?: EnvironmentSample): EnvironmentSample;
  /** Horizontal divergence of the surface current, 1/s. */
  sampleDivergence(lat: number, lon: number, timeSeconds: number): number;
}

export function emptySample(): EnvironmentSample {
  return { currentU: 0, currentV: 0, windU: 0, windV: 0, stokesU: 0, stokesV: 0, waveHeight: 0, sst: 0, landMask: 0, outOfDomain: false };
}
