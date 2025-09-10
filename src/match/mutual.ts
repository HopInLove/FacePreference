// src/match/mutual.ts
import type { Face, User, Vector } from "../types.js";
import { cosine } from "../utils/math.js";

export interface PrefConfig {
  youThresh: number; // e.g., 0.58
  themThresh: number; // e.g., 0.58
}

export interface Logistics {
  sameCity: boolean;
  languageOverlap: boolean;
  orientationOK: boolean;
  distanceOK: boolean;
}

export function prefAttraction(
  w_i: Vector | null,
  f_j: Vector | null,
  w_j: Vector | null,
  f_i: Vector | null
) {
  const safeCos = (a: Vector | null, b: Vector | null) => (a && b ? cosine(a, b) : null);
  const attr_ij = safeCos(w_i, f_j);
  const attr_ji = safeCos(w_j, f_i);
  return { attr_ij, attr_ji };
}

export function passesPreference(attr_ij: number | null, attr_ji: number | null, cfg: PrefConfig) {
  if (attr_ij == null || attr_ji == null) return false; // require both vectors
  return attr_ij >= cfg.youThresh && attr_ji >= cfg.themThresh;
}

export function passesLogistics(log: Logistics) {
  return log.orientationOK && log.languageOverlap && log.distanceOK && log.sameCity;
}

export function pairScore(attr_ij: number, attr_ji: number) {
  // bottleneck score for ordering the mutuals
  return Math.min(attr_ij, attr_ji);
}
