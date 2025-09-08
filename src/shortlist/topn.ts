import { CandidateFeatureRow, RankedCandidate, TasteVector, User, Vector } from "../types.js";
import { cosine } from "../utils/math.js";
import { mgopFromFeatures } from "../mgop/score.js";
import { Storage } from "../adapters/storage.js";

/** language overlap as Jaccard (|∩| / |∪|) */
function langOverlap(a: ("en" | "fr")[], b: ("en" | "fr")[]): number {
  const A = new Set(a),
    B = new Set(b);
  const inter = [...A].filter((x) => B.has(x)).length;
  const uni = new Set([...a, ...b]).size || 1;
  return inter / uni;
}

/** silly distance penalty for now: same neighborhood=0, different=0.2 */
function distancePenalty(neighborhoodA: string, neighborhoodB: string): number {
  return neighborhoodA === neighborhoodB ? 0 : 0.2;
}

/** placeholder schedule overlap: constant 0.6 for prototype */
function scheduleOverlap(): number {
  return 0.6;
}

/** reliability pair placeholder: constant 0.8 for prototype */
function reliabilityPair(): number {
  return 0.8;
}

/** soft dealbreaker penalty placeholder: 0 for now */
function dealbreakerPenalty(): number {
  return 0;
}

/** venue fit placeholder: 0.5 */
function venueFit(): number {
  return 0.5;
}

/** You -> Them attraction. If taste is null, return 0.5 */
function attrYouToThem(tasteVec: Vector | null, faceVec: Vector | null): number {
  if (!tasteVec || !faceVec) return 0.5;
  // map cosine [-1,1] to [0,1]
  return (cosine(tasteVec, faceVec) + 1) / 2;
}

/** Them -> You attraction.
 *  If their taste unknown, approximate with face-face similarity as a weak proxy (temporary).
 */
function attrThemToYou(
  theirTaste: Vector | null,
  yourFace: Vector | null,
  theirFace: Vector | null
): number {
  if (theirTaste && yourFace) return (cosine(theirTaste, yourFace) + 1) / 2;
  if (theirFace && yourFace) return (cosine(theirFace, yourFace) + 1) / 2; // weak proxy
  return 0.5;
}

/** Build CandidateFeatureRow list for userI against others, then compute MGOP and rank. */
export async function buildTopNForUser(
  storage: Storage,
  userI: User,
  allUsers: User[],
  tasteI: TasteVector,
  topN = 5
): Promise<RankedCandidate[]> {
  const myFace = await storage.getUserFaceAggregate(userI.userId);

  const rows: CandidateFeatureRow[] = [];
  for (const userJ of allUsers) {
    if (userJ.userId === userI.userId) continue;

    // simple orientation/age gate (prototype)
    if (userI.orientation !== userJ.orientation) continue; // crude for now
    if (userJ.age < (userI.ageMin ?? 18) || userJ.age > (userI.ageMax ?? 99)) continue;

    const theirFace = await storage.getUserFaceAggregate(userJ.userId);
    const theirTaste = (await storage.getTasteVector(userJ.userId))?.vector ?? null;

    const attrIJ = attrYouToThem(tasteI.vector, theirFace);
    const attrJI = attrThemToYou(theirTaste, myFace, theirFace);

    const row: CandidateFeatureRow = {
      userI: userI.userId,
      userJ: userJ.userId,
      attrIJ,
      attrJI,
      scheduleOverlap: scheduleOverlap(),
      distancePenalty: distancePenalty(userI.neighborhood, userJ.neighborhood),
      languageOverlap: langOverlap(userI.languages, userJ.languages),
      reliabilityPair: reliabilityPair(),
      dealbreakerPenalty: dealbreakerPenalty(),
      venueFit: venueFit(),
    };

    row.mgop = mgopFromFeatures(row);
    rows.push(row);
  }

  // sort & slice
  rows.sort((a, b) => b.mgop! - a.mgop!);
  const ranked: RankedCandidate[] = rows.slice(0, topN).map((r) => ({
    userId: r.userJ,
    mgop: r.mgop!,
    details: r,
  }));

  // store for inspection
  await storage.saveCandidateCache(userI.userId, rows);
  return ranked;
}
