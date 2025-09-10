// main.ts
import fs from "node:fs";
import path from "node:path";
import { MemoryStorage } from "./src/adapters/storage_memory.js";
import { CalibrationEvent, CalibrationPair, Face, User, Vector, TasteVector } from "./src/types.js";
import {
  buildCandidatePairs,
  selectNextPairV2,
  predictPairProb,
  type SamplerHistory,
} from "./src/calibration/pair_utils.js";
import { computeCentroidTaste } from "./src/taste/centroid.js";
import { trainRidgeTaste, stabilityCheck } from "./src/taste/ridge.js";
import { buildTopNForUser } from "./src/shortlist/topn.js";
import { cosine } from "./src/utils/math.js";

/** ---------- helpers ---------- */
async function loadJSON<T>(p: string): Promise<T> {
  const raw = await fs.promises.readFile(p, "utf8");
  return JSON.parse(raw) as T;
}

async function tryLoadUserFaces(p: string): Promise<Record<string, Vector>> {
  try {
    const raw = await fs.promises.readFile(p, "utf8");
    const obj = JSON.parse(raw);
    const out: Record<string, Vector> = {};
    if (obj && Array.isArray(obj.users)) {
      for (const u of obj.users) {
        if (u?.id && Array.isArray(u.faceVector)) out[u.id] = u.faceVector as Vector;
      }
    }
    return out;
  } catch {
    return {};
  }
}

/** Pretty console view: top-5 faces for a user's learned taste */
function showTop5YouToFaces(label: string, w: Vector | null, faces: Face[]) {
  if (!w) {
    console.log(`${label}: no taste vector`);
    return;
  }
  const scored = faces.map((f) => ({ id: f.faceId, s: cosine(w, f.vector) }));
  scored.sort((a, b) => b.s - a.s);
  const top = scored.slice(0, 5);
  console.log(
    `${label} Top-5 you→faces: ${top.map((t) => `${t.id}(${t.s.toFixed(2)})`).join(", ")}`
  );
}

/** ---------- ONE-PERSON CALIBRATION ---------- */
async function calibrateOne(
  storage: MemoryStorage,
  faces: Face[],
  users: User[],
  user: User
): Promise<Vector | null> {
  const allPairs = buildCandidatePairs(faces, 2000);
  const pairLookup = new Map<string, CalibrationPair>();
  for (const p of allPairs) pairLookup.set(p.pairId, p);

  console.log(`\nSimulating calibration for ${user.userId} ...`);

  // state
  const seenPairs = new Set<string>();
  const recentFaces = new Set<string>();
  const recentBins: number[] = [];
  const lastPairs = new Map<string, number>();
  let roundIndex = 0;

  const events: CalibrationEvent[] = [];
  const oracleTaste: Vector = user.faceVector ?? faces[0].vector;

  // training priors
  let wPrior: Vector | null = null;
  let tasteVec: Vector | null = null;

  // diagnostics
  let randomPickCnt = 0,
    totalPicks = 0,
    lastEligible = 0;

  // stop rules
  const HARD_CAP = 20;
  const MIN_EVENTS = 10;
  const STOP_MARGIN = 0.75;
  const STOP_STAB = 0.85;
  const MAX_BASE = 12;
  const MAX_EXTRA = 8;

  const shouldStop = (n: number, m: number, s: number) =>
    n >= HARD_CAP || (n >= MIN_EVENTS && m >= STOP_MARGIN && s >= STOP_STAB);

  const trainNow = (iters = 400) => {
    const lambdaNow = events.length < 8 ? 0.4 : 0.2; // a bit more reg early
    const ridge = trainRidgeTaste(user.userId, events, pairLookup, {
      lambda: lambdaNow,
      lr: 0.08,
      iters,
      w0: wPrior,
    });
    if (ridge.tv?.vector) wPrior = ridge.tv.vector;
    return ridge;
  };

  // ---- base rounds ----
  for (let round = 1; round <= MAX_BASE; round++) {
    const ridge = trainNow(400);
    tasteVec =
      ridge.tv?.vector ?? computeCentroidTaste(user.userId, events, pairLookup)?.vector ?? null;

    const pair = selectNextPairV2(
      faces,
      /* userMeta */ undefined, // keep liberal for now
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
    if (!pair) {
      console.log("No more informative pairs.");
      break;
    }

    // oracle (sim)
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

    // recency windows
    recentFaces.add(pair.a.faceId);
    recentFaces.add(pair.b.faceId);
    if (recentFaces.size > 4) {
      const arr = Array.from(recentFaces);
      recentFaces.clear();
      for (let i = 1; i < arr.length; i++) recentFaces.add(arr[i]);
    }
    const binA = (pair.a.meta as any)?.bin;
    const binB = (pair.b.meta as any)?.bin;
    if (binA !== undefined) recentBins.push(binA);
    if (binB !== undefined) recentBins.push(binB);
    while (recentBins.length > 5) recentBins.shift();

    lastPairs.set(pair.pairId, roundIndex);
    roundIndex++;

    const pPred = predictPairProb(tasteVec, pair.a, pair.b);
    console.log(
      `Round ${round}: ${pair.pairId}, oracle=${choice}, provisional P(A)=${pPred.toFixed(2)}`
    );

    const rand = allPairs[Math.floor(Math.random() * allPairs.length)];
    const debugP = predictPairProb(tasteVec, rand.a, rand.b);
    console.log(`  debug: random-pair P(A)=${(debugP ?? 0.5).toFixed(2)}`);
  }

  // ---- extras via quality gate ----
  let ridgeFinal = trainNow(500);
  let marginNow = ridgeFinal.margin;
  let stabilityNow = stabilityCheck(user.userId, events, pairLookup, {
    lambda: events.length < 8 ? 0.4 : 0.2,
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
      /* userMeta */ undefined,
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

    // recency updates
    recentFaces.add(pair.a.faceId);
    recentFaces.add(pair.b.faceId);
    if (recentFaces.size > 4) {
      const arr = Array.from(recentFaces);
      recentFaces.clear();
      for (let i = 1; i < arr.length; i++) recentFaces.add(arr[i]);
    }
    const binA = (pair.a.meta as any)?.bin;
    const binB = (pair.b.meta as any)?.bin;
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

    ridgeFinal = trainNow(500);
    marginNow = ridgeFinal.margin;
    stabilityNow = stabilityCheck(user.userId, events, pairLookup, {
      lambda: events.length < 8 ? 0.4 : 0.2,
      lr: 0.08,
      iters: 400,
      w0: wPrior,
    });

    const randomRate = totalPicks ? randomPickCnt / totalPicks : 0;
    console.log(
      `health r${String(events.length).padStart(2, "0")}: margin=${marginNow.toFixed(2)} ` +
        `stab=${stabilityNow.toFixed(2)} randomRate=${(randomRate * 100).toFixed(0)}% ` +
        `pool=${lastEligible}/${faces.length}`
    );
  }

  // ---- final taste, save, report ----
  let final: TasteVector | null =
    ridgeFinal.tv || computeCentroidTaste(user.userId, events, pairLookup) || null;

  if (!final && events.length >= 4) {
    const centroid = computeCentroidTaste(user.userId, events, pairLookup) || null;
    if (centroid) final = centroid;
  }
  if (!final) {
    console.log("Not enough data to compute taste. (Need ≥4 comparisons.)");
    return null;
  }

  await storage.saveTasteVector(final);

  console.log(
    `Final taste for ${user.userId}: method=${final.method}, sharpness=${final.sharpness.toFixed(
      3
    )}`
  );
  console.log(
    `Quality gates → margin=${marginNow.toFixed(2)} (want ≥ 0.60), stability=${stabilityNow.toFixed(
      2
    )} (want ≥ 0.90)`
  );

  return final.vector;
}

/** ---------- BOOT ---------- */
async function boot() {
  const storage = new MemoryStorage();

  // load catalog
  const faces = await loadJSON<Face[]>(path.join("data", "catalog", "faces.json"));
  storage.seedCatalog(faces);

  // load users + stitch in any user faceVectors (users_faces.json)
  const users = await loadJSON<User[]>(path.join("data", "users", "users.json"));
  for (const u of users) if (u.faceVector) storage.seedUserFace(u.userId, u.faceVector);

  const userFacesExtra = await tryLoadUserFaces(path.join("data", "users", "users_faces.json"));
  for (const u of users) {
    if (!u.faceVector && userFacesExtra[u.userId]) {
      u.faceVector = userFacesExtra[u.userId] as Vector;
      storage.seedUserFace(u.userId, u.faceVector);
    }
  }

  console.log(`Loaded ${faces.length} catalog faces and ${users.length} users.`);

  // Pick two users for the preference-first demo
  const A = users[0];
  const B = users[1] ?? users[0];

  // Calibrate both
  const wA = await calibrateOne(storage, faces, users, A);
  const wB = await calibrateOne(storage, faces, users, B);
  A.tasteVector = wA ?? null;
  B.tasteVector = wB ?? null;

  // Visual sanity checks (your taste over the catalog)
  showTop5YouToFaces(`${A.userId}`, wA ?? null, faces);
  showTop5YouToFaces(`${B.userId}`, wB ?? null, faces);

  // Mutual attraction (preference-first)
  const fA = A.faceVector ?? null;
  const fB = B.faceVector ?? null;
  const attr_ij = wA && fB ? cosine(wA, fB) : null; // A → B
  const attr_ji = wB && fA ? cosine(wB, fA) : null; // B → A

  // thresholds — tune as needed for testing
  const YOU_THRESH = 0.58;
  const THEM_THRESH = 0.58;

  const prefPass =
    attr_ij !== null && attr_ji !== null && attr_ij >= YOU_THRESH && attr_ji >= THEM_THRESH;

  console.log(`\nMutual table (preference-first):`);
  console.log(`A=${A.userId} → B=${B.userId}: attr_ij=${attr_ij?.toFixed(2) ?? "—"}`);
  console.log(`B=${B.userId} → A=${A.userId}: attr_ji=${attr_ji?.toFixed(2) ?? "—"}`);
  console.log(
    `Preference gate: ${prefPass ? "✅ PASS" : "❌ FAIL"} (thresholds ${YOU_THRESH}/${THEM_THRESH})`
  );

  // Keep the Phase-2 logistics demo (MGOP shortlist) for A if we learned a taste
  if (wA) {
    const finalForA: TasteVector = { userId: A.userId, vector: wA, method: "ridge", sharpness: 0 };
    const ranked = await buildTopNForUser(storage, A, users, finalForA, 5);
    console.log("\nTop candidates (prototype):");
    for (const r of ranked) {
      const d = r.details;
      console.log(
        `• ${r.userId}  MGOP=${r.mgop.toFixed(2)}  ` +
          `[attrIJ=${d.attrIJ.toFixed(2)}, attrJI=${d.attrJI.toFixed(2)}, ` +
          `sched=${d.scheduleOverlap.toFixed(2)}, distPen=${d.distancePenalty.toFixed(
            2
          )}, lang=${d.languageOverlap.toFixed(2)}]`
      );
    }
  }
}

boot().catch((err) => {
  console.error(err);
  process.exit(1);
});
