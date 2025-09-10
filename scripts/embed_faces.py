# scripts/embed_faces.py
import os, re, glob, json
from pathlib import Path
import numpy as np

# deps: insightface onnxruntime opencv-python-headless scikit-learn
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface import model_zoo
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

# ----------------------- Config -----------------------
CATALOG_DIR = "ingest/images/catalog"
USERS_DIR   = "ingest/images/users"   # optional per-user folders
OUT_FACES   = "data/catalog/faces.json"
OUT_USERS   = "data/users/users_faces.json"  # optional
TARGET_DIM  = 64

# Force direct embedding (skip detector) for everything in catalog
FORCE_DIRECT = True

# Recognize UTKFace filenames: age_gender_race_*.jpg
UTK_RE = re.compile(r"^(?P<age>\d{1,3})_(?P<gender>[01])_(?P<race>[0-4])_.*\.(jpg|jpeg|png)$", re.I)

# If your catalog uses subfolders by gender (e.g. catalog/f/, catalog/m/)
GENDER_FOLDERS = {"f": "f", "m": "m", "other": "other", "unk": "unknown"}

# --------------------- Helpers ------------------------
def l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def infer_gender_from_path(path: str) -> str:
    lower = path.lower()
    for k, v in GENDER_FOLDERS.items():
        if f"{os.sep}{k}{os.sep}" in lower:
            return v
    return "unknown"

def parse_utk_meta(fname: str):
    """
    From "age_gender_race_*.jpg" → meta fields.
    gender: 0=male, 1=female
    race:   0=White,1=Black,2=Asian,3=Indian,4=Other (per UTK docs)
    """
    m = UTK_RE.match(fname)
    if not m:
        return None
    age = int(m.group("age"))
    gender_code = int(m.group("gender"))
    race_code = int(m.group("race"))
    gender = "f" if gender_code == 1 else "m"
    race_map = {0: "white", 1: "black", 2: "asian", 3: "indian", 4: "other"}
    race = race_map.get(race_code, "other")
    return {"age": age, "gender": gender, "race": race}

def is_utk_file(path: str) -> bool:
    return UTK_RE.match(os.path.basename(path)) is not None

# ----------------- Model loaders -----------------
def init_face_app():
    # kept around for USERS_DIR aggregation if needed
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app

def _find_buffalo_recognition() -> str | None:
    # common default path after FaceAnalysis downloads
    cand = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
    if os.path.exists(cand):
        return cand
    # fallback: search under ~/.insightface/models for w600k_r50.onnx
    root = os.path.expanduser("~/.insightface/models")
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower() == "w600k_r50.onnx":
                return os.path.join(dirpath, fn)
    return None

def init_arcface():
    # Try the canonical registry name first, then fall back to ONNX path.
    rec = None
    try:
        rec = model_zoo.get_model("arcface_r100_v1")
    except Exception:
        rec = None
    if rec is None:
        onnx_path = _find_buffalo_recognition()
        if not onnx_path:
            raise SystemExit("ArcFace model not found. Expected ~/.insightface/models/buffalo_l/w600k_r50.onnx")
        rec = model_zoo.get_model(onnx_path)
    rec.prepare(ctx_id=-1)
    return rec

# ----------------- Embedding funcs -----------------
def embed_direct_112(rec, img_path: str, app=None):
    """
    Direct embedding (no detection): resize to 112×112 and run ArcFace ONNX.
    Handles InsightFace variants:
      - rec.get_feat(img)
      - rec.get(img)
      - rec.get(img, face)    (requires landmarks -> use app as fallback)
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] unreadable image: {img_path}")
        return None

    # Ensure 3-channel BGR and 112x112
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    aligned = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)  # keep BGR

    feat = None

    # 1) Preferred: versions that expose get_feat(aligned)
    try:
        if hasattr(rec, "get_feat"):
            feat = rec.get_feat(aligned)  # aligned is already 112×112
    except Exception as e:
        print(f"[DBG] get_feat failed for {img_path}: {e}")

    # 2) Try get(aligned) (some builds accept aligned image directly)
    if feat is None:
        try:
            feat = rec.get(aligned)
        except Exception:
            feat = None  # continue

    # 3) Try get(aligned, face) by quickly detecting a face on the aligned crop
    if feat is None and app is not None:
        try:
            faces = app.get(aligned)
            if faces:
                feat = rec.get(aligned, faces[0])
        except Exception as e:
            print(f"[DBG] get(aligned, face) failed for {img_path}: {e}")
            feat = None

    if feat is None:
        # last resort: if app found a face, you can just use its normed_embedding
        if app is not None:
            faces = app.get(aligned)
            if faces:
                emb = faces[0].normed_embedding
                return l2n(np.asarray(emb, dtype=np.float32).reshape(-1))
        return None

    emb = np.asarray(feat, dtype=np.float32).reshape(-1)
    return l2n(emb)



def embed_with_detection(app: FaceAnalysis, img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: f.det_score, reverse=True)
    return faces[0].normed_embedding  # 512-D L2-normalized

# ------------------------ Main ------------------------
def main():
    # Models
    app = init_face_app()
    rec = init_arcface()

    # recursive glob so subfolders like catalog/f/, catalog/m/ work
    cat_paths = sorted(
        p for p in glob.glob(os.path.join(CATALOG_DIR, "**", "*"), recursive=True)
        if os.path.isfile(p)
    )

    all_embs = []
    cat_recs = []

    summary = {"files": 0, "used": 0, "unreadable": 0, "no_face": 0, "deduped": 0}
    seen_ids = set()

    # 1) collect catalog embeddings (direct for all when FORCE_DIRECT)
    for i, p in enumerate(cat_paths):
        summary["files"] += 1
        fname = os.path.basename(p)

        # Choose path
        if FORCE_DIRECT:
           emb = embed_direct_112(rec, p, app)

        else:
            emb = embed_direct_112(rec, p) if is_utk_file(p) else embed_with_detection(app, p)

        if emb is None:
            summary["no_face"] += 1
            continue

        face_id = f"f{i+1:04d}"
        if face_id in seen_ids:
            summary["deduped"] += 1
            continue
        seen_ids.add(face_id)

        # Meta
        meta_utk = parse_utk_meta(fname) or {}
        gender = meta_utk.get("gender") or infer_gender_from_path(p)

        all_embs.append(emb)
        cat_recs.append({
            "faceId": face_id,
            "emb": emb,
            "gender": gender,
            "meta_utk": meta_utk
        })
        summary["used"] += 1

    # 2) optional: aggregate user folders into a mean face vector
    user_dirs = [d for d in glob.glob(os.path.join(USERS_DIR, "*")) if os.path.isdir(d)]
    user_recs = []
    for udir in user_dirs:
        uid = os.path.basename(udir)
        uvecs = []
        for ip in glob.glob(os.path.join(udir, "*")):
            emb = embed_direct_112(rec, ip) if FORCE_DIRECT or is_utk_file(ip) else embed_with_detection(app, ip)
            if emb is not None:
                uvecs.append(emb)
        if uvecs:
            umean = l2n(np.mean(np.vstack(uvecs), axis=0))
            user_recs.append({"id": uid, "emb": umean})
            all_embs.append(umean)

    if not all_embs:
        raise SystemExit("No embeddings found. Add images to ingest/images/catalog/")

    # 3) Dimensionality reduction → 64-D (robust)
    X = np.vstack(all_embs)  # (N, 512)
    N, D = X.shape
    if N >= TARGET_DIM + 1:
        pca = PCA(n_components=TARGET_DIM, svd_solver="auto", random_state=42)
        Z = pca.fit_transform(X)
    else:
        rp = GaussianRandomProjection(n_components=TARGET_DIM, random_state=42)
        Z = rp.fit_transform(X)
    Z = np.vstack([l2n(z) for z in Z])

    # 4) map back
    idx = 0
    for r in cat_recs:
        r["vec64"] = Z[idx].tolist(); idx += 1
    for r in user_recs:
        r["vec64"] = Z[idx].tolist(); idx += 1

    # 5) write faces.json with gender + UTK meta
    Path(os.path.dirname(OUT_FACES)).mkdir(parents=True, exist_ok=True)
    faces_out = []
    for r in cat_recs:
        meta = {"bin": -1, "gender": r.get("gender", "unknown")}
        utk = r.get("meta_utk", {})
        if "age" in utk: meta["age"] = int(utk["age"])
        if "race" in utk: meta["race"] = utk["race"]
        faces_out.append({"faceId": r["faceId"], "vector": r["vec64"], "meta": meta})
    with open(OUT_FACES, "w") as f:
        json.dump(faces_out, f, indent=2)
    print(f"[OK] wrote {OUT_FACES} ({len(faces_out)} faces)")

    # 6) optional users dump
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

    # 7) summary
    print(f"[SUMMARY] catalog files={summary['files']}  used={summary['used']}  "
          f"unreadable={summary['unreadable']}  no_face={summary['no_face']}  "
          f"deduped={summary['deduped']}")
    print("Direct embedding is ON (FORCE_DIRECT=True). If you want detectors again, set FORCE_DIRECT=False.")

if __name__ == "__main__":
    main()
