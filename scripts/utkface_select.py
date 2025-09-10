# scripts/utkface_select.py
import os, sys, glob, shutil
from pathlib import Path

SRC_DIR = "data/utkface"             # where you unzipped UTKFace
OUT_ROOT = "ingest/images/catalog"   # your pipeline expects this
AGE_MIN, AGE_MAX = 18, 25
MAX_PER_GENDER = 800                 # cap per gender to keep things snappy
MODE = "symlink"                     # "copy" or "symlink"

def parse_utk(filename: str):
    """
    UTKFace file pattern: age_gender_race_date&time.jpg
    e.g., 24_0_2_20170109204404335.jpg
    returns (age:int, gender:str['m'|'f'], race:int) or None if not parseable
    """
    base = os.path.basename(filename)
    root = base
    if root.endswith(".chip.jpg"):    # some variants
        root = root[:-9]
    parts = root.split("_")
    if len(parts) < 3:
        return None
    try:
        age = int(parts[0])
        gcode = int(parts[1])  # 0=male,1=female
        gender = "m" if gcode == 0 else ("f" if gcode == 1 else "unknown")
        race = int(parts[2])
        return age, gender, race
    except Exception:
        return None

def main():
    src_files = sorted([p for p in glob.glob(os.path.join(SRC_DIR, "**", "*.jpg"), recursive=True)])
    if not src_files:
        print(f"[ERR] No JPGs found under {SRC_DIR}. Did you download & unzip?")
        sys.exit(1)

    out_f = Path(OUT_ROOT, "f"); out_f.mkdir(parents=True, exist_ok=True)
    out_m = Path(OUT_ROOT, "m"); out_m.mkdir(parents=True, exist_ok=True)
    out_other = Path(OUT_ROOT, "other"); out_other.mkdir(parents=True, exist_ok=True)

    kept = {"f": 0, "m": 0, "other": 0}
    total = 0
    for p in src_files:
        meta = parse_utk(p)
        total += 1
        if not meta:
            continue
        age, gender, race = meta
        if not (AGE_MIN <= age <= AGE_MAX):
            continue
        # cap by gender
        if kept.get(gender, 0) >= MAX_PER_GENDER:
            continue

        out_dir = out_f if gender == "f" else out_m if gender == "m" else out_other
        dst = out_dir / os.path.basename(p)

        try:
            if MODE == "copy":
                shutil.copy2(p, dst)
            else:
                # create symlink; if exists, skip
                if not dst.exists():
                    os.symlink(os.path.abspath(p), dst)
        except FileExistsError:
            pass
        except OSError as e:
            # fallback to copy if symlink fails (e.g., on restrictive FS)
            shutil.copy2(p, dst)

        kept[gender] = kept.get(gender, 0) + 1

        # small safety: stop if we reached caps for m and f
        if kept["f"] >= MAX_PER_GENDER and kept["m"] >= MAX_PER_GENDER:
            break

    print(f"[SUMMARY] scanned={total} kept_f={kept['f']} kept_m={kept['m']} kept_other={kept['other']}")
    print(f"[OK] staged into {OUT_ROOT}/(f|m|other)")

if __name__ == "__main__":
    main()
