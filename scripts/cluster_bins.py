# scripts/cluster_bins.py
import json
import sys
import os
import numpy as np
from sklearn.cluster import KMeans

FACES = "data/catalog/faces.json"
DEFAULT_MAX_K = 24
SEED = 42

def choose_k(n_samples: int, max_k: int = DEFAULT_MAX_K) -> int:
    """
    Pick a sensible K for small datasets:
      - at least 1
      - at most max_k (24)
      - no more than n_samples
      - heuristic: min(max_k, max(1, int(round(np.sqrt(n_samples)))))
    """
    if n_samples <= 1:
        return 1
    heuristic = int(round(np.sqrt(n_samples)))
    k = max(1, min(max_k, n_samples, heuristic))
    return k

def main():
    with open(FACES) as f:
        faces = json.load(f)
    if not isinstance(faces, list):
        raise SystemExit("faces.json must be a list of face objects")

    X = np.array([f["vector"] for f in faces], dtype=np.float32)
    n = X.shape[0]
    if n == 0:
        raise SystemExit("No faces found in faces.json")

    # Allow override via env var: CLUSTER_MAX_K=24 (optional)
    try:
        max_k = int(os.environ.get("CLUSTER_MAX_K", DEFAULT_MAX_K))
    except ValueError:
        max_k = DEFAULT_MAX_K

    K = choose_k(n, max_k=max_k)

    if K == 1:
        # trivial assignment
        for f in faces:
            f.setdefault("meta", {})
            f["meta"]["bin"] = 0
        with open(FACES, "w") as out:
            json.dump(faces, out, indent=2)
        print(f"[OK] updated {FACES} with bin assignments (K=1) — not enough samples yet")
        return

    km = KMeans(n_clusters=K, random_state=SEED, n_init="auto").fit(X)
    labels = km.labels_.tolist()

    for f, lab in zip(faces, labels):
        f.setdefault("meta", {})
        f["meta"]["bin"] = int(lab)

    with open(FACES, "w") as out:
        json.dump(faces, out, indent=2)
    print(f"[OK] updated {FACES} with bin assignments (K={K}, n={n})")

if __name__ == "__main__":
    main()
