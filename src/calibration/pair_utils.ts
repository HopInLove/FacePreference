import { Face, CalibrationPair } from "../types.js";
import { sigmoid } from "../utils/math.js";

function pairId(a: Face, b: Face) {
  // keep deterministic ordering for id
  const [x, y] = [a.faceId, b.faceId].sort();
  return `${x}__${y}`;
}

/** Create up to maxPairs diverse pairs from the catalog. */
export function buildCandidatePairs(faces: Face[], maxPairs = 2000): CalibrationPair[] {
  const out: CalibrationPair[] = [];
  // naive: all combos; for big catalogs, sample. fine for prototype.
  for (let i = 0; i < faces.length; i++) {
    for (let j = i + 1; j < faces.length; j++) {
      const a = faces[i],
        b = faces[j];
      out.push({ pairId: pairId(a, b), a, b });
    }
  }
  // cap size if needed
  if (out.length > maxPairs) {
    // simple shuffle + trim
    for (let i = out.length - 1; i > 0; i--) {
      const k = Math.floor(Math.random() * (i + 1));
      [out[i], out[k]] = [out[k], out[i]];
    }
    return out.slice(0, maxPairs);
  }
  return out;
}

/** Predict P(pick A over B) given a taste vector; if no taste, return 0.5 */
export function predictPairProb(tasteVec: number[] | null, a: Face, b: Face): number {
  if (!tasteVec) return 0.5;
  // simple: dot(taste, a) - dot(taste, b) passed through sigmoid
  let da = 0,
    db = 0;
  for (let i = 0; i < tasteVec.length; i++) {
    da += tasteVec[i] * a.vector[i];
    db += tasteVec[i] * b.vector[i];
  }
  return sigmoid(da - db);
}

/** Choose next pair by uncertainty (closest to 0.5) with light diversity guard by bins. */
export function selectNextPair(
  pairs: CalibrationPair[],
  tasteVec: number[] | null,
  seenPairIds: Set<string>,
  recentlyUsedFaceIds: Set<string>,
  recentBins: number[] // queue of last bins used
): CalibrationPair | null {
  // 1) collect top-K most uncertain among unseen & not-recent faces
  const K = 400; // small candidate pool
  type Row = { p: CalibrationPair; uncert: number; binA?: number; binB?: number };
  const cand: Row[] = [];

  for (const p of pairs) {
    if (seenPairIds.has(p.pairId)) continue;
    if (recentlyUsedFaceIds.has(p.a.faceId) || recentlyUsedFaceIds.has(p.b.faceId)) continue;

    const prob = predictPairProb(tasteVec, p.a, p.b);
    const uncertainty = Math.abs(prob - 0.5);
    const binA = (p.a.meta as any)?.bin as number | undefined;
    const binB = (p.b.meta as any)?.bin as number | undefined;

    cand.push({ p, uncert: uncertainty, binA, binB });
  }

  if (cand.length === 0) return null;

  // sort by uncertainty ASC (closest to 0.5 first) and keep top-K
  cand.sort((a, b) => a.uncert - b.uncert);
  const top = cand.slice(0, Math.min(K, cand.length));

  // 2) among top-K, pick the one with the best diversity score
  let best: CalibrationPair | null = null;
  let bestScore = Infinity;
  for (const row of top) {
    const usedA = row.binA !== undefined && recentBins.includes(row.binA);
    const usedB = row.binB !== undefined && recentBins.includes(row.binB);
    // stronger nudge to alternate bins
    const diversityPenalty = (usedA ? 0.2 : 0) + (usedB ? 0.2 : 0);
    const score = row.uncert + diversityPenalty;
    if (score < bestScore) {
      bestScore = score;
      best = row.p;
    }
  }

  return best;
}
