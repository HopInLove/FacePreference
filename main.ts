// main.ts
import fs from "node:fs";
import path from "node:path";
import { MemoryStorage } from "./src/adapters/storage_memory.js";
import { CalibrationEvent, CalibrationPair, Face, User, Vector } from "./src/types.js";
import {
  // keep builder for pairLookup, but selection will use V2
  buildCandidatePairs,
  selectNextPairV2,
  predictPairProb,
  type UserMeta,
  type SamplerHistory,
} from "./src/calibration/pair_utils.js";
import { computeCentroidTaste } from "./src/taste/centroid.js";
import { trainRidgeTaste, stabilityCheck } from "./src/taste/ridge.js";
import { buildTopNForUser } from "./src/shortlist/topn.js";
import { sigmoid } from "./src/utils/math.js";

async function loadJSON<T>(p: string): Promise<T> {
  const raw = await fs.promises.readFile(p, "utf8");
  return JSON.parse(raw) as T;
}

async function boot() {
  const storage = new MemoryStorage();

  // ---- load data ----
  const faces = await loadJSON<Face[]>(path.join("data", "catalog", "faces.json"));
  storage.seedCatalog(faces);

  const users = await loadJSON<User[]>(path.join("data", "users", "users.json"));
  for (const u of users) if (u.faceVector) storage.seedUserFace(u.userId, u.faceVector);

  console.log(`Loaded ${faces.length} catalog faces and ${users.length} users.`);

  // Build full pair space once (for ridge training lookups)
  const allPairs = buildCandidatePairs(faces, 2000);
  const pairLookup = new Map<string, CalibrationPair>();
  for (const p of allPairs) pairLookup.set(p.pairId, p);

  const user = users[0];
  console.log(`Simulating calibration for ${user.userId} ...`);

  // Orientation for calibration (default to "m->f" if missing)
  const userMeta: UserMeta = { orientation: (user as any)?.meta?.orientation ?? "m->f" };

  // ---- calibration state ----
  const seenPairs = new Set<string>();
  const recentFaces = new Set<string>(); // small rolling set
  const recentBins: number[] = []; // small rolling queue
  const lastPairs = new Map<string, number>(); // pairId -> last round index
  let roundIndex = 0;

  const events: CalibrationEvent[] = [];
  const oracleTaste: Vector = user.faceVector ?? faces[0].vector;

  // Warm-start prior (reduces jitter)
  let wPrior: Vector | null = null;
  let tasteVec: Vector | null = null;

  // health metrics
  let randomPickCnt = 0,
    totalPicks = 0;
  let lastEligible = 0;

  // stop rules (adaptive)
  const HARD_CAP = 20,
    MIN_EVENTS = 10,
    STOP_MARGIN = 0.75,
    STOP_STAB = 0.85;
  const MAX_BASE = 12; // initial rounds
  const MAX_EXTRA = 8; // allow a couple more if gates not met

  function shouldStop(evCount: number, margin: number, stability: number) {
    if (evCount >= HARD_CAP) return true;
    if (evCount < MIN_EVENTS) return false;
    return margin >= STOP_MARGIN && stability >= STOP_STAB;
  }

  function logHealth(
    evCount: number,
    margin: number,
    stability: number,
    eligible: number,
    total: number
  ) {
    const randomRate = totalPicks ? randomPickCnt / totalPicks : 0;
    console.log(
      `health r${String(evCount).padStart(2, "0")}: ` +
        `margin=${margin.toFixed(2)} stab=${stability.toFixed(2)} ` +
        `randomRate=${(randomRate * 100).toFixed(0)}% pool=${eligible}/${total}`
    );
  }

  // dev oracle using user's faceVector (fallback random)
  function oracleChoice(a: Face, b: Face): "A" | "B" {
    const w = oracleTaste;
    if (!w) return Math.random() < 0.5 ? "A" : "B";
    let da = 0,
      db = 0;
    for (let i = 0; i < w.length; i++) {
      da += w[i] * a.vector[i];
      db += w[i] * b.vector[i];
    }
    return da >= db ? "A" : "B";
  }

  // ---- base rounds ----
  for (let round = 1; round <= MAX_BASE; round++) {
    // train (warm-started ridge)
    const ridge = trainRidgeTaste(user.userId, events, pairLookup, {
      lambda: 0.2,
      lr: 0.08,
      iters: 400,
      w0: wPrior,
    });
    if (ridge.tv?.vector) wPrior = ridge.tv.vector;

    tasteVec =
      ridge.tv?.vector ?? computeCentroidTaste(user.userId, events, pairLookup)?.vector ?? null;

    // pick next pair with orientation-aware sampler (V2)
    const pair = selectNextPairV2(
      faces,
      userMeta,
      tasteVec,
      {
        recentBins,
        recentlyUsedFaceIds: recentFaces,
        seenPairIds: seenPairs,
        lastPairs,
        roundIndex,
      } as SamplerHistory,
      {
        exploreRate: 0.15,
        recentBinCooldown: 3,
        pairReuseCooldown: 6,
        maxPairs: 2000,
        onDiag: (d) => {
          totalPicks++;
          // treat low-uncertainty (<0.2) or “no taste yet” as random-ish for monitoring
          if ((d.pickedUncertainty ?? 0) < 0.2 || tasteVec === null) randomPickCnt++;
          lastEligible = d.pool.eligible;
          console.log(
            `sampler: considered=${d.considered} topK=${d.topK} ` +
              `picked=${d.pickedPairId} u=${d.pickedUncertainty?.toFixed(2)} ` +
              `pool(elig=${d.pool.eligible}/${d.pool.total})`
          );
        },
      }
    );

    if (!pair) {
      console.log("No more informative pairs.");
      break;
    }

    const pA_oracle = predictPairProb(oracleTaste, pair.a, pair.b);
    const choice = pA_oracle >= 0.5 ? "A" : "B";

    // record event
    const ev: CalibrationEvent = {
      userId: user.userId,
      pairId: pair.pairId,
      choice,
      ts: Date.now(),
    };
    events.push(ev);
    await storage.saveCalibrationEvent(ev);
    seenPairs.add(pair.pairId);

    // maintain recent faces (cooldown window ~4 ids)
    recentFaces.add(pair.a.faceId);
    recentFaces.add(pair.b.faceId);
    if (recentFaces.size > 4) {
      const arr = Array.from(recentFaces);
      recentFaces.clear();
      for (let i = 1; i < arr.length; i++) recentFaces.add(arr[i]);
    }

    // maintain recent bins (queue len ~5)
    const binA = (pair.a.meta as any)?.bin as number | undefined;
    const binB = (pair.b.meta as any)?.bin as number | undefined;
    if (binA !== undefined) recentBins.push(binA);
    if (binB !== undefined) recentBins.push(binB);
    while (recentBins.length > 5) recentBins.shift();

    // reuse cooldown tracking
    lastPairs.set(pair.pairId, roundIndex);
    roundIndex++;

    const pPred = predictPairProb(tasteVec, pair.a, pair.b);
    console.log(
      `Round ${round}: ${pair.pairId}, oracle=${choice}, provisional P(A)=${pPred.toFixed(2)}`
    );

    // DEBUG: random probe (unchanged from your version)
    const rand = allPairs[Math.floor(Math.random() * allPairs.length)];
    const debugP = predictPairProb(tasteVec, rand.a, rand.b);
    console.log(`  debug: random-pair P(A)=${(debugP ?? 0.5).toFixed(2)}`);
  }

  // ---- quality gate: allow extras if needed (adaptive) ----
  let ridgeFinal = trainRidgeTaste(user.userId, events, pairLookup, {
    lambda: 0.2,
    lr: 0.08,
    iters: 500,
    w0: wPrior,
  });
  if (ridgeFinal.tv?.vector) wPrior = ridgeFinal.tv.vector;

  let marginNow = ridgeFinal.margin;
  let stabilityNow = stabilityCheck(user.userId, events, pairLookup, {
    lambda: 0.2,
    lr: 0.08,
    iters: 400,
    w0: wPrior,
  });

  let extra = 0;
  while (!shouldStop(events.length, marginNow, stabilityNow) && extra < MAX_EXTRA) {
    tasteVec =
      ridgeFinal.tv?.vector ??
      computeCentroidTaste(user.userId, events, pairLookup)?.vector ??
      null;

    const pair = selectNextPairV2(
      faces,
      userMeta,
      tasteVec,
      {
        recentBins,
        recentlyUsedFaceIds: recentFaces,
        seenPairIds: seenPairs,
        lastPairs,
        roundIndex,
      } as SamplerHistory,
      {
        exploreRate: 0.15,
        recentBinCooldown: 3,
        pairReuseCooldown: 6,
        maxPairs: 2000,
        onDiag: (d) => {
          totalPicks++;
          if ((d.pickedUncertainty ?? 0) < 0.2 || tasteVec === null) randomPickCnt++;
          lastEligible = d.pool.eligible;
          console.log(
            `sampler: considered=${d.considered} topK=${d.topK} ` +
              `picked=${d.pickedPairId} u=${d.pickedUncertainty?.toFixed(2)} ` +
              `pool(elig=${d.pool.eligible}/${d.pool.total})`
          );
        },
      }
    );

    if (!pair) break;

    const pA_oracle = predictPairProb(oracleTaste, pair.a, pair.b);
    const choice = pA_oracle >= 0.5 ? "A" : "B";

    const ev: CalibrationEvent = {
      userId: user.userId,
      pairId: pair.pairId,
      choice,
      ts: Date.now(),
    };
    events.push(ev);
    await storage.saveCalibrationEvent(ev);
    seenPairs.add(pair.pairId);

    // recency maintenance
    recentFaces.add(pair.a.faceId);
    recentFaces.add(pair.b.faceId);
    if (recentFaces.size > 4) {
      const arr = Array.from(recentFaces);
      recentFaces.clear();
      for (let i = 1; i < arr.length; i++) recentFaces.add(arr[i]);
    }
    const binA = (pair.a.meta as any)?.bin as number | undefined;
    const binB = (pair.b.meta as any)?.bin as number | undefined;
    if (binA !== undefined) recentBins.push(binA);
    if (binB !== undefined) recentBins.push(binB);
    while (recentBins.length > 5) recentBins.shift();

    lastPairs.set(pair.pairId, roundIndex);
    roundIndex++;

    extra++;
    const pPred = predictPairProb(tasteVec, pair.a, pair.b);
    console.log(
      `Extra ${extra}: ${pair.pairId}, oracle=${choice}, provisional P(A)=${pPred.toFixed(2)}`
    );

    // retrain & re-check after each extra
    ridgeFinal = trainRidgeTaste(user.userId, events, pairLookup, {
      lambda: 0.2,
      lr: 0.08,
      iters: 500,
      w0: wPrior,
    });
    if (ridgeFinal.tv?.vector) wPrior = ridgeFinal.tv.vector;

    marginNow = ridgeFinal.margin;
    stabilityNow = stabilityCheck(user.userId, events, pairLookup, {
      lambda: 0.2,
      lr: 0.08,
      iters: 400,
      w0: wPrior,
    });

    // round health
    const randomRate = totalPicks ? randomPickCnt / totalPicks : 0;
    console.log(
      `health r${String(events.length).padStart(2, "0")}: ` +
        `margin=${marginNow.toFixed(2)} stab=${stabilityNow.toFixed(2)} ` +
        `randomRate=${(randomRate * 100).toFixed(0)}% pool=${lastEligible}/${faces.length}`
    );
  }

  // ---- final taste, save, report ----
  const final = ridgeFinal.tv ?? computeCentroidTaste(user.userId, events, pairLookup);
  if (!final) {
    console.log("Not enough data to compute taste. (Need ≥4 comparisons.)");
    return;
  }
  await storage.saveTasteVector(final);

  console.log(
    `Final taste for ${user.userId}: method=${final.method}, sharpness=${final.sharpness.toFixed(3)}`
  );
  console.log(
    `Quality gates → margin=${marginNow.toFixed(2)} (want ≥ 0.60), ` +
      `stability=${stabilityNow.toFixed(2)} (want ≥ 0.90)`
  );

  // ---- Top-N candidates (prototype) ----
  const ranked = await buildTopNForUser(storage, user, users, final, 5);
  console.log("\nTop candidates (prototype):");
  for (const r of ranked) {
    const d = r.details;
    console.log(
      `• ${r.userId}  MGOP=${r.mgop.toFixed(2)}  ` +
        `[attrIJ=${d.attrIJ.toFixed(2)}, attrJI=${d.attrJI.toFixed(2)}, ` +
        `sched=${d.scheduleOverlap.toFixed(2)}, distPen=${d.distancePenalty.toFixed(2)}, ` +
        `lang=${d.languageOverlap.toFixed(2)}]`
    );
  }
}

boot().catch((err) => {
  console.error(err);
  process.exit(1);
});
