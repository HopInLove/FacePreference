import fs from "node:fs";
import path from "node:path";
import { MemoryStorage } from "./src/adapters/storage_memory.js";
import { Face, User } from "./src/types.js";

async function loadJSON<T>(p: string): Promise<T> {
  const raw = await fs.promises.readFile(p, "utf8");
  return JSON.parse(raw) as T;
}

async function boot() {
  const storage = new MemoryStorage();

  const faces = await loadJSON<Face[]>(path.join("data", "catalog", "faces.json"));
  storage.seedCatalog(faces);

  const users = await loadJSON<User[]>(path.join("data", "users", "users.json"));
  for (const u of users) {
    if (u.faceVector) storage.seedUserFace(u.userId, u.faceVector);
  }

  console.log(`Loaded ${faces.length} catalog faces and ${users.length} users.`);
  console.log("✅ Scaffold OK. Next: calibration → taste → Top-N.");
}

boot().catch((err) => {
  console.error(err);
  process.exit(1);
});
