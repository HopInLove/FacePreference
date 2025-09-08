import { CandidateFeatureRow } from "../types.js";
import { sigmoid } from "../utils/math.js";

/** Hand-tuned weights v1 — we’ll calibrate later. */
const W = {
  bias: -0.5,
  attrIJ: 1.8,
  attrJI: 1.4,
  scheduleOverlap: 1.2,
  distancePenalty: -0.9,
  languageOverlap: 0.5,
  reliabilityPair: 0.7,
  dealbreakerPenalty: -2.0,
  venueFit: 0.3,
};

export function mgopFromFeatures(r: CandidateFeatureRow): number {
  const z =
    W.bias +
    W.attrIJ * r.attrIJ +
    W.attrJI * r.attrJI +
    W.scheduleOverlap * r.scheduleOverlap +
    W.distancePenalty * r.distancePenalty +
    W.languageOverlap * r.languageOverlap +
    W.reliabilityPair * r.reliabilityPair +
    W.dealbreakerPenalty * r.dealbreakerPenalty +
    W.venueFit * r.venueFit;

  return sigmoid(z);
}
