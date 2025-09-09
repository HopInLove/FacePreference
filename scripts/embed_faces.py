# scripts/embed_faces.py
import os, glob, json
from pathlib import Path
import numpy as np

# pip install insightface onnxruntime opencv-python-headless scikit-learn numpy pillow
import insightface
from insightface.app import FaceAnalysis
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

CATALOG_DIR = "ingest/images/catalog"
USERS_DIR   = "ingest/images/users"   # optional
OUT_FACES   = "data/catalog/faces.json"
OUT_USERS   = "data/users/users.json" # optional
TARGET_DIM  = 64   # keep your current dim

# ---- gender folders → meta.gender ----
GENDER_FOLDERS = {"f": "f", "m": "m", "other": "other", "unk": "unknown"}

def infer_gender_from_path(path: str) -> str:
    # infer from path segments .../catalog/<f|m|other|unk>/file.jpg
    parts = [p.lower() for p in Path(path).parts]
    for k, v in GENDER_FOLDERS.items():
        if k in parts:
            return v
    return "unknown"

# ---- utils ----
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".JPG", ".PNG", ".JPEG", ".WEBP")
DEDUP_SIM_THRESHOLD = 0.98

def l2n(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def read_image_any(path):
    # Try OpenCV first (fast), then Pillow → numpy (RGB→BGR for insightface)
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        pil = Image.open(path).convert("RGB")
        arr = np.asarray(pil)[:, :, ::-1].copy()
        return arr
    except Exception:
        return None

def best_emb(app, img_path):
    img = read_image_any(img_path)
    if img is None:
        print(f"[WARN] unreadable image: {img_path}")
        return None
    faces = app.get(img)
    if not faces:
        print(f"[WARN] no face detected: {img_path}")
        return None
    faces.sort(key=lambda f: f.det_score, reverse=True)
    return faces[0].normed_embedding  # 512-D L2-normalized

def main():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640,640))  # CPU

    # recursive glob under catalog (supports subfolders f/, m/, other/, unk/)
    cat_paths = sorted(
        [p for p in glob.glob(os.path.join(CATALOG_DIR, "**", "*"), recursive=True)
         if os.path.isfile(p) and p.lower().endswith(SUPPORTED_EXTS)]
    )

    all_embs = []
    cat_recs = []
    used, unreadable, noface, duped = 0, 0, 0, 0

    # 1) collect catalog embeddings (with dedupe + gender tag)
    for i, p in enumerate(cat_paths):
        img = read_image_any(p)
        if img is None:
            print(f"[WARN] unreadable image: {p}")
            unreadable += 1
            continue

        faces = app.get(img)
        if not faces:
            print(f"[WARN] no face detected: {p}")
            noface += 1
            continue

        faces.sort(key=lambda f: f.det_score, reverse=True)
        emb = faces[0].normed_embedding  # 512-D

        # de-duplicate vs kept embeddings
        if any(cosine(emb, e) > DEDUP_SIM_THRESHOLD for e in all_embs):
            print(f"[INFO] skip near-duplicate: {p}")
            duped += 1
            continue

        gender = infer_gender_from_path(p)
        all_embs.append(emb)
        cat_recs.append({"faceId": f"f{i+1:04d}", "emb": emb, "gender": gender})
        used += 1

    print(f"[SUMMARY] catalog files={len(cat_paths)}  used={used}  unreadable={unreadable}  no_face={noface}  deduped={duped}")

    # optional: users -> robust mean of multiple selfies
    user_dirs = [d for d in glob.glob(os.path.join(USERS_DIR, "*")) if os.path.isdir(d)]
    user_recs = []
    for udir in user_dirs:
        uid = os.path.basename(udir)
        uvecs = []
        for ip in glob.glob(os.path.join(udir, "*")):
            if not os.path.isfile(ip) or not ip.lower().endswith(SUPPORTED_EXTS):
                continue
            emb = best_emb(app, ip)
            if emb is not None:
                uvecs.append(emb)
        if uvecs:
            umean = l2n(np.mean(np.vstack(uvecs), axis=0))
            user_recs.append({"id": uid, "emb": umean})
            all_embs.append(umean)

    if not all_embs:
        raise SystemExit("No embeddings found. Add images to ingest/images/catalog/")

    # --- robust dimensionality reduction to 64-D ---
    X = np.vstack(all_embs)  # (N, 512)
    N, D = X.shape

    if N >= TARGET_DIM + 1:
        # Enough samples: proper PCA→64
        pca = PCA(n_components=TARGET_DIM, svd_solver="auto", random_state=42)
        Z = pca.fit_transform(X)
    else:
        # Few samples: Johnson–Lindenstrauss random projection to 64-D
        rp = GaussianRandomProjection(n_components=TARGET_DIM, random_state=42)
        Z = rp.fit_transform(X)

    # L2-normalize each row
    Z = np.vstack([l2n(z) for z in Z])
    # --- end robust block ---

    # map back to records (first |cat_recs| rows = catalog, then users)
    idx = 0
    for r in cat_recs:
        r["vec64"] = Z[idx].tolist(); idx += 1
    for r in user_recs:
        r["vec64"] = Z[idx].tolist(); idx += 1

    # write faces.json in existing array-of-objects shape, with gender + bin placeholder
    Path(os.path.dirname(OUT_FACES)).mkdir(parents=True, exist_ok=True)
    faces_out = [
        {
            "faceId": r["faceId"],
            "vector": r["vec64"],
            "meta": {"bin": -1, "gender": r.get("gender", "unknown")}
        }
        for r in cat_recs
    ]
    with open(OUT_FACES, "w") as f:
        json.dump(faces_out, f, indent=2)
    print(f"[OK] wrote {OUT_FACES} ({len(faces_out)} faces)")

    # optional: users.json (no gender here; orientation lives on users later)
    if user_recs:
        Path(os.path.dirname(OUT_USERS)).mkdir(parents=True, exist_ok=True)
        users_out = {
            "users": [
                {"id": r["id"], "faceVector": r["vec64"], "meta": {}}
                for r in user_recs
            ]
        }
        with open(OUT_USERS, "w") as f:
            json.dump(users_out, f, indent=2)
        print(f"[OK] wrote {OUT_USERS} ({len(users_out['users'])} users)")

if __name__ == "__main__":
    main()
