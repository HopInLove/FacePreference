import { Vector } from "../types.js";

export function dot(a: Vector, b: Vector): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
export function zeros(dim: number): Vector {
  return new Array(dim).fill(0);
}
export function addScaled(out: Vector, x: Vector, alpha: number): void {
  for (let i = 0; i < out.length; i++) out[i] += alpha * x[i];
}
export function copy(v: Vector): Vector {
  return v.slice();
}
export function l2norm(v: Vector): number {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
}
export function normalize(v: Vector): Vector {
  const n = l2norm(v);
  return v.map((x) => x / n);
}
export function sub(a: Vector, b: Vector): Vector {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
export function cosine(a: Vector, b: Vector): number {
  let dotp = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dotp += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb) || 1;
  return dotp / denom;
}
