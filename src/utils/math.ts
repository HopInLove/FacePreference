import { Vector } from "../types.js";

export function cosine(a: Vector, b: Vector): number {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb) || 1;
  return dot / denom;
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
