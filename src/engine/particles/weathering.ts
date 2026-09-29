// Mass loss per particle, forward runs only. Same formulation as the reference slick engine; every coefficient
// is a tunable assumption (oil type, temperature response), not a calibrated value.
//   evaporation (Fingas, log form):  F(age) = (A + B·SST)·ln(1 + age_min),  F ≤ F_max,   dm = −m0·ΔF
//   natural dispersion (first order): dm = −m·(1 − exp(−k·dt))
import type { EnvironmentProvider, EnvironmentSample } from '../env/types';
import { STATUS_FLOATING, type Particles } from './particles';

export interface WeatheringParams {
  weathering: boolean;
  evapA: number; // dimensionless per ln(min)
  evapB: number; // per °C per ln(min)
  evapMax: number; // volatile fraction
  dispersionRate: number; // s⁻¹
}

export const DEFAULT_WEATHERING: WeatheringParams = { weathering: true, evapA: 0.025, evapB: 0.00045, evapMax: 0.45, dispersionRate: 1.5e-6 };

/** Apply one step to floating particles whose age has already been advanced by dt (> 0). Returns kg removed. */
export function weather(P: Particles, env: EnvironmentProvider, t: number, dt: number, w: WeatheringParams, smp: EnvironmentSample): { evaporatedKg: number; dispersedKg: number } {
  let ev = 0, di = 0;
  if (!w.weathering || !(dt > 0)) return { evaporatedKg: 0, dispersedKg: 0 };
  const keep = Math.exp(-w.dispersionRate * dt);
  for (let i = 0; i < P.count; i++) {
    if (P.status[i] !== STATUS_FLOATING || P.mass[i] <= 0) continue;
    const a1 = P.ageSeconds[i], a0 = a1 - dt;
    const rate = Math.max(0, w.evapA + w.evapB * env.sample(P.lat[i], P.lon[i], t, smp).sst);
    const f = Math.min(w.evapMax, P.evaporated[i] + rate * (Math.log1p(a1 / 60) - Math.log1p(Math.max(0, a0) / 60)));
    const e = Math.min(P.mass[i], P.initialMass[i] * (f - P.evaporated[i]));
    P.evaporated[i] = f;
    P.mass[i] -= e;
    const d = P.mass[i] * (1 - keep);
    P.mass[i] -= d;
    ev += e;
    di += d;
  }
  return { evaporatedKg: ev, dispersedKg: di };
}
