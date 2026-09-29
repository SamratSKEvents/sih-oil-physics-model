// Geodesy on a spherical Earth. Pure TS, no Cesium.
// All vectors are (east, north) in metres or m/s. Angles in degrees unless named *Rad.

export const EARTH_RADIUS_M = 6371008.8; // mean Earth radius (IUGG), metres
export const DEG = Math.PI / 180;
const MAX_LAT = 89.9; // guard: cos(lat) -> 0 near the poles
const MIN_COS = Math.cos(MAX_LAT * DEG);

/** Wrap longitude into [-180, 180). */
export function wrapLon(lon: number): number {
  if (lon >= -180 && lon < 180) return lon;
  return ((((lon + 180) % 360) + 360) % 360) - 180;
}

export function clampLat(lat: number): number {
  return lat > MAX_LAT ? MAX_LAT : lat < -MAX_LAT ? -MAX_LAT : lat;
}

/** cos(lat) guarded away from zero. */
export function safeCos(lat: number): number {
  const c = Math.cos(lat * DEG);
  return c < MIN_COS ? MIN_COS : c;
}

/** Signed longitude difference b - a in degrees, shortest way round, in [-180, 180). */
export function dLon(a: number, b: number): number {
  return wrapLon(b - a);
}

/**
 * Time-derivative of position for a velocity (u east, v north) in m/s:
 *   dlat/dt = v / R,   dlon/dt = u / (R cos lat)   (degrees per second)
 */
export function latRate(v: number): number {
  return v / EARTH_RADIUS_M / DEG;
}
export function lonRate(u: number, lat: number): number {
  return u / (EARTH_RADIUS_M * safeCos(lat)) / DEG;
}

/** Displace a point by (dx east, dy north) metres:  Δlat = dy/R,  Δlon = dx/(R cos lat). */
export function displace(lat: number, lon: number, dx: number, dy: number, out: { lat: number; lon: number }): void {
  const nlat = clampLat(lat + dy / EARTH_RADIUS_M / DEG);
  out.lat = nlat;
  out.lon = wrapLon(lon + dx / (EARTH_RADIUS_M * safeCos(lat)) / DEG);
}

/** Local equirectangular projection metres (east, north) of (lat, lon) relative to (lat0, lon0). */
export function toLocal(lat0: number, lon0: number, lat: number, lon: number, cos0 = safeCos(lat0)): [number, number] {
  return [dLon(lon0, lon) * DEG * EARTH_RADIUS_M * cos0, (lat - lat0) * DEG * EARTH_RADIUS_M];
}

export function fromLocal(lat0: number, lon0: number, x: number, y: number, cos0 = safeCos(lat0)): [number, number] {
  return [clampLat(lat0 + y / EARTH_RADIUS_M / DEG), wrapLon(lon0 + x / (EARTH_RADIUS_M * cos0) / DEG)];
}

/** Short-range distance in metres (local equirectangular about the mean latitude). */
export function localDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const c = safeCos((lat1 + lat2) / 2);
  const dx = dLon(lon1, lon2) * DEG * EARTH_RADIUS_M * c;
  const dy = (lat2 - lat1) * DEG * EARTH_RADIUS_M;
  return Math.hypot(dx, dy);
}
