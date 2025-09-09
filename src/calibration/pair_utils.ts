// src/calibration/pair_utils.ts
import { Face, CalibrationPair } from "../types.js";
import { sigmoid } from "../utils/math.js";

/** ---------- Types & helpers ---------- */

export type Gender = "f" | "m" | "other" | "unknown";
export type Orientation = "m->f" | "f->m" | "m->m" | "f->f" | "bi";

export interface UserMeta {
  orientation?: Orientation;
  // ... extend as needed (age, langs, etc.)
}

export interface SamplerOptions {
  /** show some ineligible faces to avoid overfitting */
  exploreRate?: number; // default 0.15
  /** avoid reusing a recently seen bin */
  recentBinCooldown?: number; // default 3
  /** avoid reusing a face back-to-back */
  faceCooldown?: number; // default 1 (handled by caller's recentlyUsedFaceIds)
  /** allow a pair to reappear after this many rounds */
  pairReuseCooldown?: number; // default 6
  /** cap pairspace for perf */
  maxPairs?: number; // default 2000
  /** optional debug sink */
  onDiag?: (d: SamplerDiagnostics) => void;
}

export interface SamplerHistory {
  /** queue of previous bins (latest at end) */
  recentBins: number[];
  /** faces we don't want to use immediately again */
  recentlyUsedFaceIds: Set<string>;
  /** pairs already presented this session */
  seenPairIds: Set<string>;
  /** last round index a pair was shown (by pairId) */
  lastPairs?: Map<string, number>;
  /** current round index (0-based) */
  roundIndex?: number;
}

export interface SamplerDiagnostics {
  considered: number;
  topK: number;
  pickedPairId?: string;
  pickedUncertainty?: number;
  penalties?: {
    diversityPenalty: number;
    reusePenalty: number;
    facePenalty: number;
  };
  pool: {
    total: number;
    eligible: number;
    explored: number;
  };
}

function pairId(a: Face, b: Face) {
  // deterministic
  const [x, y] = [a.faceId, b.faceId].sort();
  return `${x}__${y}`;
}

function faceEligibleFor(user: UserMeta | undefined, face: Face): boolean {
  if (!user?.orientation) return true; // default liberal if no orientation
  const g = ((face.meta as any)?.gender as Gender | undefined) ?? "unknown";
  const o = user.orientation;
  if (o === "bi") return true;
  if (o === "m->f") return g === "f" || g === "unknown";
  if (o === "f->m") return g === "m" || g === "unknown";
  if (o === "m->m") return g === "m" || g === "unknown";
  if (o === "f->f") return g === "f" || g === "unknown";
  return true;
}

/** ---------- Building pairs ---------- */

export function buildCandidatePairs(faces: Face[], maxPairs = 2000): CalibrationPair[] {
  // Backward-compatible naive builder (all combos, capped)
  const out: CalibrationPair[] = [];
  for (let i = 0; i < faces.length; i++) {
    for (let j = i + 1; j < faces.length; j++) {
      const a = faces[i],
        b = faces[j];
      out.push({ pairId: pairId(a, b), a, b });
    }
  }
  if (out.length > maxPairs) {
    for (let i = out.length - 1; i > 0; i--) {
      const k = Math.floor(Math.random() * (i + 1));
      [out[i], out[k]] = [out[k], out[i]];
    }
    return out.slice(0, maxPairs);
  }
  return out;
}

/** New: orientation-aware pool with exploration */
export function buildCandidatePairsV2(
  faces: Face[],
  user?: UserMeta,
  opts: SamplerOptions = {}
): CalibrationPair[] {
  const { exploreRate = 0.15, maxPairs = 2000 } = opts;

  const eligible = faces.filter((f) => faceEligibleFor(user, f));
  const ineligible = faces.filter((f) => !faceEligibleFor(user, f));

  const exploreCount = Math.min(ineligible.length, Math.ceil(eligible.length * exploreRate));
  // sample a small explore slice
  for (let i = ineligible.length - 1; i > 0; i--) {
    const k = Math.floor(Math.random() * (i + 1));
    [ineligible[i], ineligible[k]] = [ineligible[k], ineligible[i]];
  }
  const pool = [...eligible, ...ineligible.slice(0, exploreCount)];

  // now build pairs on the pool (capped)
  return buildCandidatePairs(pool, maxPairs);
}

/** ---------- Scoring & selection ---------- */

/** Predict P(pick A over B) given taste; if no taste yet, return 0.5 */
export function predictPairProb(tasteVec: number[] | null, a: Face, b: Face): number {
  if (!tasteVec) return 0.5;
  // dot(taste, a) - dot(taste, b) → sigmoid
  let da = 0,
    db = 0;
  for (let i = 0; i < tasteVec.length; i++) {
    da += tasteVec[i] * a.vector[i];
    db += tasteVec[i] * b.vector[i];
  }
  return sigmoid(da - db);
}

function uncertaintyFromProb(p: number): number {
  // 1 at 0.5; 0 at 0 or 1
  return 1 - Math.abs(p - 0.5) * 2;
}

function diversityPenaltyFor(a: Face, b: Face, recentBins: number[], cooldown: number): number {
  const binA = (a.meta as any)?.bin as number | undefined;
  const binB = (b.meta as any)?.bin as number | undefined;

  const window = recentBins.slice(-cooldown);
  const usedA = binA !== undefined && window.includes(binA);
  const usedB = binB !== undefined && window.includes(binB);

  // tuned light penalties; adjust if you see repetition
  return (usedA ? 0.15 : 0) + (usedB ? 0.15 : 0);
}

function pairReusePenalty(a: Face, b: Face, hist: SamplerHistory, cooldown: number): number {
  if (!hist.lastPairs || hist.roundIndex === undefined) return 0;
  const key1 = pairId(a, b);
  const last = hist.lastPairs.get(key1);
  if (last === undefined) return 0;
  const gap = hist.roundIndex - last;
  return gap < cooldown ? 0.4 : 0; // block until cooled down
}

function faceReusePenalty(a: Face, b: Face, recentlyUsedFaceIds: Set<string>): number {
  let p = 0;
  if (recentlyUsedFaceIds.has(a.faceId)) p += 0.25;
  if (recentlyUsedFaceIds.has(b.faceId)) p += 0.25;
  return p;
}

/**
 * New selection with orientation-aware pool, uncertainty-first ranking,
 * and penalties for low diversity / recent reuse.
 *
 * Returns the chosen pair and optionally emits diagnostics via opts.onDiag.
 */
export function selectNextPairV2(
  faces: Face[],
  user: UserMeta | undefined,
  tasteVec: number[] | null,
  hist: SamplerHistory,
  opts: SamplerOptions = {}
): CalibrationPair | null {
  const {
    exploreRate = 0.15,
    recentBinCooldown = 3,
    pairReuseCooldown = 6,
    maxPairs = 2000,
    onDiag,
  } = opts;

  // Build orientation-aware candidate pairs
  const pairs = buildCandidatePairsV2(faces, user, { exploreRate, maxPairs });

  // Filter out seen pairs & face cooldown set
  const rows: {
    p: CalibrationPair;
    u: number; // uncertainty (1 high, 0 low)
    divPen: number;
    reusePen: number;
    facePen: number;
  }[] = [];

  for (const p of pairs) {
    if (hist.seenPairIds?.has(p.pairId)) continue;
    if (hist.recentlyUsedFaceIds?.has(p.a.faceId) || hist.recentlyUsedFaceIds?.has(p.b.faceId))
      continue;

    const prob = predictPairProb(tasteVec, p.a, p.b);
    const u = uncertaintyFromProb(prob);
    const divPen = diversityPenaltyFor(p.a, p.b, hist.recentBins ?? [], recentBinCooldown);
    const reusePen = pairReusePenalty(p.a, p.b, hist, pairReuseCooldown);
    const facePen = faceReusePenalty(p.a, p.b, hist.recentlyUsedFaceIds ?? new Set());

    rows.push({ p, u, divPen, reusePen, facePen });
  }

  if (rows.length === 0) {
    onDiag?.({
      considered: 0,
      topK: 0,
      pool: {
        total: faces.length,
        eligible: faces.filter((f) => faceEligibleFor(user, f)).length,
        explored: 0,
      },
    });
    return null;
  }

  // rank by (uncertainty - total penalty), highest first
  rows.sort(
    (A, B) =>
      B.u - (B.divPen + B.reusePen + B.facePen) - (A.u - (A.divPen + A.reusePen + A.facePen))
  );

  // soft top-K to avoid always picking the same best
  const K = Math.min(400, rows.length);
  const top = rows.slice(0, K);

  // pick the best among top-K (already sorted)
  const best = top[0];

  onDiag?.({
    considered: rows.length,
    topK: K,
    pickedPairId: best.p.pairId,
    pickedUncertainty: best.u,
    penalties: {
      diversityPenalty: best.divPen,
      reusePenalty: best.reusePen,
      facePenalty: best.facePen,
    },
    pool: {
      total: faces.length,
      eligible: faces.filter((f) => faceEligibleFor(user, f)).length,
      explored: Math.max(
        0,
        Math.min(
          faces.length - faces.filter((f) => faceEligibleFor(user, f)).length,
          Math.ceil(
            faces.filter((f) => faceEligibleFor(user, f)).length * (opts.exploreRate ?? 0.15)
          )
        )
      ),
    },
  });

  return best.p;
}

/** ---------- Backward-compatible wrapper ----------
 * Keeps your existing call sites working.
 * Uses liberal defaults and no orientation if you don’t pass a user.
 */
export function selectNextPair(
  pairs: CalibrationPair[],
  tasteVec: number[] | null,
  seenPairIds: Set<string>,
  recentlyUsedFaceIds: Set<string>,
  recentBins: number[]
): CalibrationPair | null {
  // Reconstruct minimal "faces" + "userless" context to call V2.
  // We’ll just rescore the provided pairs.
  const hist: SamplerHistory = {
    recentBins,
    recentlyUsedFaceIds,
    seenPairIds,
  };

  // Build rows from provided pairs
  const rows: {
    p: CalibrationPair;
    u: number;
    divPen: number;
    facePen: number;
  }[] = [];

  for (const p of pairs) {
    if (seenPairIds.has(p.pairId)) continue;
    if (recentlyUsedFaceIds.has(p.a.faceId) || recentlyUsedFaceIds.has(p.b.faceId)) continue;

    const prob = predictPairProb(tasteVec, p.a, p.b);
    const u = uncertaintyFromProb(prob);
    const divPen = diversityPenaltyFor(p.a, p.b, recentBins ?? [], 3);
    const facePen = faceReusePenalty(p.a, p.b, recentlyUsedFaceIds);

    rows.push({ p, u, divPen, facePen });
  }

  if (rows.length === 0) return null;

  rows.sort((A, B) => B.u - (B.divPen + B.facePen) - (A.u - (A.divPen + A.facePen)));
  const K = Math.min(400, rows.length);
  return rows.slice(0, K)[0].p;
}
