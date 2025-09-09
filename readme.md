# FacePreference (attraction-core)

End-to-end prototype for facial preference calibration → taste vector → Top-N shortlist → match score (MGOP).

Runs DB-free with synthetic data to validate the loop before wiring Postgres/Supabase and real embeddings.

Clean storage seam so we can swap the backend later with minimal code changes.

# Quick start

 Node 18+ recommended

npm install

 (optional) regenerate synthetic data

npx tsx scripts/generate_synthetic.ts

 run the prototype

npm run dev

You’ll see:

12 base calibration rounds (+ up to 6 extra if quality gates ask),

final taste vector stats (margin, stability),

a Top-N list with feature breakdown and the combined MGOP score.

# What this prototype proves

A small set of A vs B taps learns a personal taste vector (“your type”).

The next pairs are chosen smartly (uncertainty + diversity) to learn fast.

Candidates can be ranked using two-sided attraction + logistics (MGOP).

The loop runs in <~150ms for a few hundred candidates on a laptop.

# Project structure
.
├── data/ # inputs for the run
│ ├── catalog/faces.json # reference faces (vectors + bin labels)
│ └── users/users.json # mock users (faceVector + metadata)
├── scripts/
│ └── generate_synthetic.ts # creates synthetic faces/users for prototyping
├── src/
│ ├── adapters/
│ │ ├── storage.ts # Storage interface (DB-agnostic seam)
│ │ └── storage_memory.ts # In-memory storage (prototype)
│ ├── calibration/
│ │ └── pair_utils.ts # build pairs, uncertainty scoring, diversity
│ ├── embeddings/ # (empty) real face model adapter will live here
│ ├── mgop/
│ │ └── score.ts # MGOP v1 (blend of features → sigmoid)
│ ├── shortlist/
│ │ └── topn.ts # filters + attraction + features → Top-N
│ ├── taste/
│ │ ├── centroid.ts # centroid fallback + μ helper
│ │ └── ridge.ts # pairwise ridge ranker + quality gates + stability
│ ├── types.ts # shared types (Face, User, etc.)
│ └── utils/math.ts # small vector math helpers
├── main.ts # wires everything together for a demo run
└── tsconfig.json

# How it works (plain English)

1. Calibration (12–16 taps, A vs B)

We show two reference faces; the user taps one (or skip). The system picks pairs it’s least sure about and mixes diverse “bins” so we learn broadly.

2. Taste vector (“your type”)

From taps we learn a small number-vector:

Default: pairwise logistic ranker (ridge-regularized).

Few taps: centroid difference = average(liked) − average(disliked).

Quality gates:

Margin (are choices easy to explain?)

Stability (even vs odd taps similar?)
If weak, ask 2–4 more taps.

3. Shortlist (Top-N)

We score candidates with two-sided attraction (you→them via cosine(taste, face), and them→you if available) plus practical fit:

schedule overlap (placeholder in prototype),

distance penalty (neighborhood),

language overlap (EN/FR),

reliability (placeholder).

These feed a simple MGOP formula (linear weights → sigmoid), then we sort and print Top-N.

In this prototype, “faces” and “users” are synthetic vectors—no real images. The loop, math, and future APIs are the same we’ll use with real embeddings.

# Commands & data
Generate synthetic data
npx tsx scripts/generate_synthetic.ts

Produces:

data/catalog/faces.json – ~720 face vectors labeled into ~24 bins

data/users/users.json – ~60 mock users with metadata

Run
npm run dev

Reads the JSON files, runs a full calibration session for the first user, and prints Top-N.

# Storage seam (swap DB later)

All core code talks to a tiny Storage interface (src/adapters/storage.ts).
Today: storage_memory.ts.
Later: add storage_supabase.ts (or direct Postgres) without changing business logic.

# Frontend heads-up 

Planned endpoints (names may change; flow won’t):

GET /v1/attraction/pairs → return 1–2 preloaded pairs to show now

POST /v1/attraction/choice → { pairId, choice: 'A'|'B'|'skip' } (idempotent)

GET /v1/matches/candidates → Top-N with feature breakdown (transparency)

POST /v1/matches/{id}/confirm|cancel|outcome → scheduling & feedback loop (later)

UI tips:

small progress bar (“learning your type…”),

skip option, instant tap feedback,

image preloading for snappy UX,

offline queueing of taps.

# Swapping in real face embeddings (later — no rewrite)

Add a face model adapter under src/embeddings/ (e.g., InsightFace via ONNX/TensorRT).

Replace synthetic JSON with vectors from the model:

Catalog: ingest reference images → detect & align → embed → k-means to bins → write faces.json.
User selfies: 2–3 photos → quality filter (face present, low blur) → robust mean → save vector.

The rest (calibration, taste, shortlist, MGOP) stays identical.

# Key files to skim first

main.ts – demo run (full loop)

src/taste/ridge.ts – learn the taste vector + quality gates

src/calibration/pair_utils.ts – pick the next pair (uncertain + diverse)

src/shortlist/topn.ts – compute Top-N and MGOP components

# FAQ (short)

Are we learning a global “hotness” score?
No. Only per-user taste. Final ranking also includes logistics & reliability.

Will users get stuck seeing one “look”?
No. We use diversity bonuses, exploration slots, and never hard-filter by looks.

What’s MGOP?
A combined match-go probability from simple features (two-sided attraction, schedule, distance, language, reliability). Starts as logistic; we’ll calibrate on real outcomes later.

Privacy in production?
Store vectors, not images (or short-retain encrypted images), require consent, encrypt at rest, audit access. We’ll follow local rules when we move beyond R&D.

# Next steps

Replace placeholder schedule overlap with availability windows.

Add Supabase/Postgres adapter implementing Storage.

Add real embeddings adapter + catalog builder (keep file shapes the same).

Calibrate MGOP on closed beta outcomes (mutual confirm + show-up).

# Troubleshooting

“No more pairs to show” too early → regenerate synthetic data (scripts/generate_synthetic.ts), or increase maxPairs in pair_utils.ts.

Stability low → (common on synthetic) ensure even/odd split is enabled; increase bin-diversity penalty; allow a couple extra rounds.

Perf issues → limit candidate pool (M) in topn.ts before MGOP.

# (Optional) Real-world pilot — not done yet

Status: not implemented yet. This is the plan when we’re ready to test with ~20 consenting adults (2–3 photos each).
You won’t rewrite code—just generate real faces.json / users.json and re-run the existing loop.

0. Montreal/Québec guardrails (must-do)

Explicit consent for biometrics (what we collect, why, retention, deletion).

Under Québec’s Law 25 / IT Act, biometric systems must be declared to the CAI ~60 days before deployment if you create a centralized biometric database.

Canada (PIPEDA/OPC) treats biometrics as highly sensitive → meaningful consent, privacy impact assessment, data minimization.

To avoid a declaration during early R&D, keep embeddings local/ephemeral. If you persist vectors centrally, assume CAI rules apply.

This is product guidance, not legal advice.

1. Data to collect (pilot)

Participants: ~20 adults, written consent.

Per person: 2–3 selfies (frontal-ish, good light, single face).

Policy: store vectors only; delete images post-embedding; retention 30–60 days; encrypt at rest; least-privilege access; audit reads.

2. Pilot folder layout
   real_dataset/
   catalog/
   p001/ _.jpg
   p002/ _.jpg
   ...
   users/
   u_001/ _.jpg
   u_002/ _.jpg
   ...

catalog/ → “reference faces” used in A-vs-B.

users/ → pilot users (or teammates) for end-to-end testing.

3. Generate real embeddings (self-hosted)

Easiest: a small Python script (InsightFace + ONNX, CPU is fine) that writes the same JSON your TS code already reads.

Install (Python 3.10+):

python -m venv .venv && source .venv/bin/activate
pip install insightface onnxruntime opencv-python-headless scikit-learn numpy

Script path: scripts/gen_embeddings.py (add this file), then:

python scripts/gen_embeddings.py

# outputs:

# data/catalog/faces.json

# data/users/users.json

✔️ The TS code accepts any vector dimension (64/512/etc).
✔️ “bin” labels come from k-means so your diversity logic still works.

4. Run the pipeline as-is
   npm run dev

Expect P(A) ~ 0.50 on shown pairs (uncertainty sampling).

Watch margin & stability; real images usually help stability.

If low quality hurts detection, raise threshold (e.g., det_score >= 0.7) and re-embed.

5. Minimum privacy hygiene (pilot)

Delete raw images after embedding (or keep only on device).

Store vectors only, encrypted at rest.

Auto-delete pilot vectors after 30–60 days unless participants re-consent.

Least-privilege access; log every read.

One-pager consent (purpose, data, retention, access, withdrawal).

<details> <summary><strong>Appendix: example <code>gen_embeddings.py</code></strong> (click to expand)</summary>
# See README text in the repo for the full script body you can paste into scripts/gen_embeddings.py.
# It uses InsightFace (buffalo_l), embeds catalog/users, runs k-means for bins,
# and writes data/catalog/faces.json + data/users/users.json.
# (Omitted here for brevity.)

</details>
