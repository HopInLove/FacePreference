import fs from "node:fs";
import path from "node:path";
import { MemoryStorage } from "./src/adapters/storage_memory.js";
import { CalibrationEvent, CalibrationPair, Face, User, Vector } from "./src/types.js";
import {
  buildCandidatePairs,
  selectNextPair,
  predictPairProb,
} from "./src/calibration/pair_utils.js";
import { computeCentroidTaste } from "./src/taste/centroid.js";
import { trainRidgeTaste, stabilityCheck } from "./src/taste/ridge.js";
import { buildTopNForUser } from "./src/shortlist/topn.js";

async function loadJSON<T>(p: string): Promise<T> {
  const raw = await fs.promises.readFile(p, "utf8");
  return JSON.parse(raw) as T;
}

async function boot() {
  const storage = new MemoryStorage();

  const faces = await loadJSON<Face[]>(path.join("data", "catalog", "faces.json"));
  storage.seedCatalog(faces);
  const users = await loadJSON<User[]>(path.join("data", "users", "users.json"));
  for (const u of users) if (u.faceVector) storage.seedUserFace(u.userId, u.faceVector);

  console.log(`Loaded ${faces.length} catalog faces and ${users.length} users.`);

  const allPairs = buildCandidatePairs(faces, 2000);
  const pairLookup = new Map<string, CalibrationPair>();
  for (const p of allPairs) pairLookup.set(p.pairId, p);

  const user = users[0];
  console.log(`Simulating calibration for ${user.userId} ...`);

  const seenPairs = new Set<string>();
  const recentFaces = new Set<string>();
  const recentBins: number[] = []; // keep last ~5 bins for diversity
  const events: CalibrationEvent[] = [];

  const oracleTaste = user.faceVector ?? faces[0].vector;

  // Warm-start prior for ridge to reduce jitter
  let wPrior: Vector | null = null;

  const MAX_BASE = 12; // was 10
  const MAX_EXTRA = 6; // was 4

  // ---- base rounds ----
  for (let round = 1; round <= MAX_BASE; round++) {
    const ridge = trainRidgeTaste(user.userId, events, pairLookup, {
      lambda: 0.2,
      lr: 0.08,
      iters: 400,
      w0: wPrior,
    });
    if (ridge.tv?.vector) wPrior = ridge.tv.vector;

    const tasteVec =
      ridge.tv?.vector ?? computeCentroidTaste(user.userId, events, pairLookup)?.vector ?? null;

    const next = selectNextPair(allPairs, tasteVec, seenPairs, recentFaces, recentBins);
    if (!next) {
      console.log("No more pairs to show.");
      break;
    }

    const pA = predictPairProb(oracleTaste, next.a, next.b);
    const choice = pA >= 0.5 ? "A" : "B";

    const ev: CalibrationEvent = {
      userId: user.userId,
      pairId: next.pairId,
      choice,
      ts: Date.now(),
    };
    events.push(ev);
    await storage.saveCalibrationEvent(ev);
    seenPairs.add(next.pairId);

    // maintain recent faces
    recentFaces.add(next.a.faceId);
    recentFaces.add(next.b.faceId);
    if (recentFaces.size > 4) {
      const arr = Array.from(recentFaces);
      recentFaces.clear();
      for (let i = 1; i < arr.length; i++) recentFaces.add(arr[i]);
    }

    // maintain recent bins (diversity)
    const binA = (next.a.meta as any)?.bin as number | undefined;
    const binB = (next.b.meta as any)?.bin as number | undefined;
    if (binA !== undefined) recentBins.push(binA);
    if (binB !== undefined) recentBins.push(binB);
    while (recentBins.length > 5) recentBins.shift(); // keep last 5

    const pPred = predictPairProb(tasteVec, next.a, next.b);
    console.log(
      `Round ${round}: ${next.pairId}, oracle=${choice}, provisional P(A)=${pPred.toFixed(2)}`
    );

    // DEBUG: random pair probe should drift away from 0.50 as taste sharpens
    const rand = allPairs[Math.floor(Math.random() * allPairs.length)];
    const debugP = predictPairProb(tasteVec, rand.a, rand.b);
    console.log(`  debug: random-pair P(A)=${(debugP ?? 0.5).toFixed(2)}`);
  }

  // ---- quality gate: ask for up to 6 extra pairs if needed ----
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
  while ((marginNow < 0.6 || stabilityNow < 0.9) && extra < MAX_EXTRA) {
    const tasteVec =
      ridgeFinal.tv?.vector ??
      computeCentroidTaste(user.userId, events, pairLookup)?.vector ??
      null;

    const next = selectNextPair(allPairs, tasteVec, seenPairs, recentFaces, recentBins);
    if (!next) break;

    const pA = predictPairProb(oracleTaste, next.a, next.b);
    const choice = pA >= 0.5 ? "A" : "B";

    const ev: CalibrationEvent = {
      userId: user.userId,
      pairId: next.pairId,
      choice,
      ts: Date.now(),
    };
    events.push(ev);
    await storage.saveCalibrationEvent(ev);
    seenPairs.add(next.pairId);

    // recency maintenance
    recentFaces.add(next.a.faceId);
    recentFaces.add(next.b.faceId);
    if (recentFaces.size > 4) {
      const arr = Array.from(recentFaces);
      recentFaces.clear();
      for (let i = 1; i < arr.length; i++) recentFaces.add(arr[i]);
    }
    const binA = (next.a.meta as any)?.bin as number | undefined;
    const binB = (next.b.meta as any)?.bin as number | undefined;
    if (binA !== undefined) recentBins.push(binA);
    if (binB !== undefined) recentBins.push(binB);
    while (recentBins.length > 5) recentBins.shift();

    extra++;
    const pPred = predictPairProb(tasteVec, next.a, next.b);
    console.log(
      `Extra ${extra}: ${next.pairId}, oracle=${choice}, provisional P(A)=${pPred.toFixed(2)}`
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
  }

  // ---- final taste, save, report ----
  let final = ridgeFinal.tv ?? computeCentroidTaste(user.userId, events, pairLookup);
  if (!final) {
    console.log("Not enough data to compute taste. (Need ≥4 comparisons.)");
    return;
  }
  await storage.saveTasteVector(final);

  console.log(
    `Final taste for ${user.userId}: method=${final.method}, sharpness=${final.sharpness.toFixed(3)}`
  );
  console.log(
    `Quality gates → margin=${marginNow.toFixed(2)} (want ≥ 0.60), stability=${stabilityNow.toFixed(2)} (want ≥ 0.90)`
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
