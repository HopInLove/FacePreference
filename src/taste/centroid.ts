import { CalibrationEvent, CalibrationPair, TasteVector, Vector } from "../types.js";
import { normalize, sub } from "../utils/math.js";

/** Build a taste vector using robust-ish centroid difference. */
export function computeCentroidTaste(
  userId: string,
  events: CalibrationEvent[],
  pairLookup: Map<string, CalibrationPair>
): TasteVector | null {
  const liked: Vector[] = [];
  const disliked: Vector[] = [];
  for (const ev of events) {
    if (ev.choice === "skip") continue;
    const pair = pairLookup.get(ev.pairId);
    if (!pair) continue;
    const pickA = ev.choice === "A";
    const vA = pair.a.vector;
    const vB = pair.b.vector;
    if (pickA) {
      liked.push(vA);
      disliked.push(vB);
    } else {
      liked.push(vB);
      disliked.push(vA);
    }
  }
  if (liked.length < 2 || disliked.length < 2) return null;

  const dim = liked[0].length;
  const mean = (xs: Vector[]) => {
    const m = new Array(dim).fill(0);
    for (const v of xs) for (let i = 0; i < dim; i++) m[i] += v[i];
    for (let i = 0; i < dim; i++) m[i] /= xs.length;
    return m;
  };

  const muLike = mean(liked);
  const muDis = mean(disliked);
  const raw = sub(muLike, muDis);
  const vec = normalize(raw);
  const sharpness = Math.sqrt(raw.reduce((s, x) => s + x * x, 0));
  return { userId, vector: vec, sharpness, method: "centroid" };
}

/** μ helper for sign alignment: mean(liked) − mean(disliked). */
export function likedMinusDislikedDelta(
  events: CalibrationEvent[],
  pairLookup: Map<string, CalibrationPair>
): Vector | null {
  const liked: Vector[] = [];
  const disliked: Vector[] = [];
  for (const ev of events) {
    if (ev.choice === "skip") continue;
    const pair = pairLookup.get(ev.pairId);
    if (!pair) continue;
    const pickA = ev.choice === "A";
    const vA = pair.a.vector,
      vB = pair.b.vector;
    if (pickA) {
      liked.push(vA);
      disliked.push(vB);
    } else {
      liked.push(vB);
      disliked.push(vA);
    }
  }
  if (liked.length === 0 || disliked.length === 0) return null;

  const dim = liked[0].length;
  const mean = (xs: Vector[]) => {
    const m = new Array(dim).fill(0);
    for (const v of xs) for (let i = 0; i < dim; i++) m[i] += v[i];
    for (let i = 0; i < dim; i++) m[i] /= xs.length;
    return m;
  };

  const muLike = mean(liked);
  const muDis = mean(disliked);
  const mu = new Array(dim).fill(0);
  for (let i = 0; i < dim; i++) mu[i] = muLike[i] - muDis[i];
  return mu;
}
