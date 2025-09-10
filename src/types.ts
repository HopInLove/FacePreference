// src/types.ts
export type Vector = number[];

export interface Face {
  faceId: string;
  vector: Vector;
  meta?: Record<string, unknown>;
}

export interface User {
  userId: string;
  age: number;
  orientation: "straight" | "gay" | "bi" | "other";
  languages: ("en" | "fr")[];
  neighborhood: string;
  maxTravelMins: number;
  faceVector?: Vector;
  ageMin?: number;
  ageMax?: number;
  hardNoSmokers?: boolean;
  /** Learned from calibration (A-vs-B). May be null pre-calibration. */
  tasteVector?: Vector | null;
}

export interface CalibrationPair {
  pairId: string;
  a: Face;
  b: Face;
}

export type Choice = "A" | "B" | "skip";

export interface CalibrationEvent {
  userId: string;
  pairId: string;
  choice: Choice;
  ts: number;
}

export interface TasteVector {
  userId: string;
  vector: Vector;
  sharpness: number;
  method: "ridge" | "centroid";
}

export interface CandidateFeatureRow {
  userI: string;
  userJ: string;
  attrIJ: number;
  attrJI: number;
  scheduleOverlap: number;
  distancePenalty: number;
  languageOverlap: number;
  reliabilityPair: number;
  dealbreakerPenalty: number;
  venueFit: number;
  mgop?: number;
}

export interface RankedCandidate {
  userId: string;
  mgop: number;
  details: CandidateFeatureRow;
}
