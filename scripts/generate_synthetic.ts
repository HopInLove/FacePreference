// scripts/generate_synthetic.ts
// Generates a synthetic face catalog (vectors + bins) and mock users.
// Writes to: data/catalog/faces.json and data/users/users.json

import fs from "node:fs/promises";
import path from "node:path";

type Vector = number[];

type FaceRow = {
  faceId: string;
  vector: Vector;
  meta: { bin: number };
};

type UserRow = {
  userId: string;
  age: number;
  orientation: "straight" | "gay" | "bi";
  languages: ("en" | "fr")[];
  neighborhood: string;
  maxTravelMins: number;
  faceVector: Vector; // aggregate
  ageMin?: number;
  ageMax?: number;
};

function randn() {
  // Box–Muller
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function normalize(v: Vector): Vector {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / n);
}

function addNoise(v: Vector, sigma: number): Vector {
  const out = v.slice();
  for (let i = 0; i < out.length; i++) out[i] += sigma * randn();
  return out;
}

function zeros(dim: number): Vector {
  return new Array(dim).fill(0);
}

function lincomb(a: Vector, b: Vector, wa: number, wb: number): Vector {
  const out = a.slice();
  for (let i = 0; i < out.length; i++) out[i] = wa * a[i] + wb * b[i];
  return out;
}

async function main() {
  // Tunables — you can change these later without touching code elsewhere
  const DIM = Number(process.env.DIM || 64);
  const BINS = Number(process.env.BINS || 24);
  const FACES_PER_BIN = Number(process.env.FACES_PER_BIN || 30); // ~720 faces total
  const USERS = Number(process.env.USERS || 60);
  const FACE_NOISE = Number(process.env.FACE_NOISE || 0.25);
  const USER_NOISE = Number(process.env.USER_NOISE || 0.2);

  // 1) Make diversity-bin centroids
  const centroids: Vector[] = [];
  for (let b = 0; b < BINS; b++) {
    const c = Array.from({ length: DIM }, () => randn());
    centroids.push(normalize(c));
  }

  // 2) Generate faces around centroids
  const faces: FaceRow[] = [];
  let faceCounter = 1;
  for (let b = 0; b < BINS; b++) {
    for (let k = 0; k < FACES_PER_BIN; k++) {
      const base = centroids[b];
      const noisy = normalize(addNoise(base, FACE_NOISE));
      faces.push({
        faceId: `f${String(faceCounter).padStart(4, "0")}`,
        vector: noisy,
        meta: { bin: b },
      });
      faceCounter++;
    }
  }

  // 3) Mock users: pick 1–2 favorite bins, blend centroids + noise for their faceVector
  const neighborhoods = ["Plateau", "Griffintown", "Downtown", "NDG", "Mile End", "Rosemont"];
  const users: UserRow[] = [];
  for (let u = 0; u < USERS; u++) {
    const id = `u_${String(u + 1).padStart(3, "0")}`;
    const age = 20 + Math.floor(Math.random() * 12); // 20–31
    const orientationPool: UserRow["orientation"][] = ["straight", "gay", "bi"];
    const orientation = orientationPool[Math.floor(Math.random() * orientationPool.length)];

    // Lang skew: ~60% bilingual, ~25% EN, ~15% FR
    const r = Math.random();
    let languages: ("en" | "fr")[] = [];
    if (r < 0.6) languages = ["en", "fr"];
    else if (r < 0.85) languages = ["en"];
    else languages = ["fr"];

    const neighborhood = neighborhoods[Math.floor(Math.random() * neighborhoods.length)];
    const maxTravelMins = [20, 25, 30, 35, 40][Math.floor(Math.random() * 5)];

    // choose primary & secondary bins to hint their "look"
    const b1 = Math.floor(Math.random() * BINS);
    const b2 = Math.floor(Math.random() * BINS);
    const w1 = 0.7 + 0.2 * Math.random(); // 0.7–0.9
    const w2 = 1 - w1;

    const baseVec = normalize(lincomb(centroids[b1], centroids[b2], w1, w2));
    const faceVector = normalize(addNoise(baseVec, USER_NOISE));

    const user: UserRow = {
      userId: id,
      age,
      orientation,
      languages,
      neighborhood,
      maxTravelMins,
      faceVector,
      ageMin: 20,
      ageMax: 31,
    };
    users.push(user);
  }

  // 4) Save
  const facesPath = path.join("data", "catalog", "faces.json");
  const usersPath = path.join("data", "users", "users.json");

  await fs.mkdir(path.dirname(facesPath), { recursive: true });
  await fs.mkdir(path.dirname(usersPath), { recursive: true });

  await fs.writeFile(facesPath, JSON.stringify(faces, null, 2), "utf8");
  await fs.writeFile(usersPath, JSON.stringify(users, null, 2), "utf8");

  console.log(
    `✅ generated ${faces.length} faces across ${BINS} bins (dim=${DIM}) and ${users.length} users.\n` +
      `   wrote:\n   - ${facesPath}\n   - ${usersPath}`
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
