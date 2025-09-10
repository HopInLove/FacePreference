import fs from "node:fs";

const mainPath = "data/users/users.json";
const facesPath = "data/users/users_faces.json";

const main = JSON.parse(fs.readFileSync(mainPath, "utf8")); // array of users
const facesBlob = JSON.parse(fs.readFileSync(facesPath, "utf8")); // { users: [ {id, faceVector}, ... ] }
const byId = new Map(facesBlob.users.map((u) => [u.id, u.faceVector]));

const merged = main.map((u) => {
  const fv = byId.get(u.userId);
  return fv ? { ...u, faceVector: fv } : u;
});

fs.writeFileSync(mainPath, JSON.stringify(merged, null, 2));
console.log(`[OK] merged ${byId.size} face vectors into ${mainPath}`);
