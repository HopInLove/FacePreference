import { CalibrationEvent, CalibrationPair, TasteVector, Vector } from "../types.js";
import { dot, sigmoid, zeros, addScaled, copy, normalize, l2norm } from "../utils/math.js";
import { likedMinusDislikedDelta } from "./centroid.js";

/** Options for per-user ridge logistic ranker. */
export interface RidgeOpts {
  lambda?: number; // L2 strength toward w0
  lr?: number; // learning rate
  iters?: number; // gradient steps
  w0?: Vector | null; // global prior (same dim as face vectors)
}

/** Build dataset: x = (faceA - faceB), y = +1 if picked A else -1. */
function buildPairwiseDataset(
  events: CalibrationEvent[],
  pairLookup: Map<string, CalibrationPair>
): { X: Vector[]; Y: number[]; dim: number } {
  const X: Vector[] = [];
  const Y: number[] = [];
  let dim = -1;

  for (const ev of events) {
    if (ev.choice === "skip") continue;
    const pair = pairLookup.get(ev.pairId);
    if (!pair) continue;
    const vA = pair.a.vector;
    const vB = pair.b.vector;
    if (dim < 0) dim = vA.length;
    const x = new Array(dim);
    for (let i = 0; i < dim; i++) x[i] = vA[i] - vB[i];
    const y = ev.choice === "A" ? +1 : -1;
    X.push(x);
    Y.push(y);
  }
  return { X, Y, dim: dim < 0 ? 0 : dim };
}

/** Train tiny logistic regression on pairwise diffs with L2 toward w0. */
export function trainRidgeTaste(
  userId: string,
  events: CalibrationEvent[],
  pairLookup: Map<string, CalibrationPair>,
  opts: RidgeOpts = {}
): { tv: TasteVector | null; margin: number } {
  const { X, Y, dim } = buildPairwiseDataset(events, pairLookup);
  if (dim === 0 || X.length < 4) return { tv: null, margin: 0 };

  const lambda = opts.lambda ?? 0.1;
  const lr = opts.lr ?? 0.1;
  const iters = opts.iters ?? 200;
  const w0 = opts.w0 ?? zeros(dim);
  let w: Vector = copy(w0);

  // batch gradient descent
  const n = X.length;
  for (let t = 0; t < iters; t++) {
    const grad = zeros(dim);
    // data term
    for (let i = 0; i < n; i++) {
      const xi = X[i];
      const yi = Y[i]; // +1 or -1
      const z = dot(w, xi); // yi * z goes into logistic
      const s = 1 / (1 + Math.exp(yi * z)); // σ(-yi*z)
      // grad += - yi * xi * σ(-yi*z)
      addScaled(grad, xi, -yi * s);
    }
    // L2 toward w0
    for (let j = 0; j < dim; j++) grad[j] += lambda * (w[j] - w0[j]);
    // update
    for (let j = 0; j < dim; j++) w[j] -= (lr / n) * grad[j];
  }

  // --- sign alignment using μ = mean(liked) − mean(disliked) ---
  const mu = likedMinusDislikedDelta(events, pairLookup);
  if (mu) {
    let s = 0;
    for (let i = 0; i < mu.length; i++) s += w[i] * mu[i];
    if (s < 0) {
      for (let i = 0; i < w.length; i++) w[i] = -w[i];
    }
  }

  // quality: margin = mean σ( yi * (w·xi) )
  let margin = 0;
  for (let i = 0; i < X.length; i++) {
    const yz = Y[i] * dot(w, X[i]);
    margin += sigmoid(yz);
  }
  margin /= X.length;

  const vec = normalize(w);
  const sharpness = l2norm(w.map((v, i) => v - w0[i]));

  return { tv: { userId, vector: vec, sharpness, method: "ridge" }, margin };
}

/** Stability: split events, train on halves, align both with μ, compare cosine. */
export function stabilityCheck(
  userId: string,
  events: CalibrationEvent[],
  pairLookup: Map<string, CalibrationPair>,
  opts: RidgeOpts = {}
): number {
  if (events.length < 6) return 0;

  // Even/odd split = more balanced than first-half vs second-half
  const aEvents = events.filter((_, i) => i % 2 === 0);
  const bEvents = events.filter((_, i) => i % 2 === 1);
  if (aEvents.length < 3 || bEvents.length < 3) return 0;

  const left = trainRidgeTaste(userId, aEvents, pairLookup, opts).tv;
  const right = trainRidgeTaste(userId, bEvents, pairLookup, opts).tv;
  if (!left || !right) return 0;

  // Align both to μ so sign flips don't hurt stability
  const mu = likedMinusDislikedDelta(events, pairLookup);
  const align = (v: Vector): Vector => {
    if (!mu) return v;
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * mu[i];
    return s < 0 ? v.map((x) => -x) : v;
  };
  const va = align(left.vector),
    vb = align(right.vector);

  let dotp = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < va.length; i++) {
    dotp += va[i] * vb[i];
    na += va[i] * va[i];
    nb += vb[i] * vb[i];
  }
  return dotp / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}
