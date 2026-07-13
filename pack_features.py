"""Pack the 17,988 per-clip .npy feature files into ONE memmap file on Easha,
so training does 1 file open + paged reads instead of 17,988 small-file reads
(far gentler on the NTFS driver). Padded to 128 frames; real lengths saved
alongside. Adds a `pack_idx` column to the manifest for positional lookup.

Re-runnable: writes to a .tmp file and atomically renames on success.
"""
import os, time
import numpy as np
import pandas as pd

ARCHIVE = "/Volumes/Easha/SSD Backup/Bornil_Dataset/archive"
BASE = "/Volumes/Easha/SSD Backup/Bornil_Dataset"
FEATURE_DIR = os.path.join(BASE, "features_mediapipe_v2")
MANIFEST = os.path.join(ARCHIVE, "processed_manifest.csv")
OUT = os.path.join(BASE, "features_packed_v2.npy")
LEN_OUT = os.path.join(BASE, "features_packed_v2_lengths.npy")
MAXF, DIM = 128, 258

man = pd.read_csv(MANIFEST)
N = len(man)
print(f"packing {N} clips -> {OUT} (shape ({N},{MAXF},{DIM}) float32 ~{N*MAXF*DIM*4/1e9:.2f} GB)", flush=True)

arr = np.lib.format.open_memmap(OUT + ".tmp", mode="w+", dtype=np.float32, shape=(N, MAXF, DIM))
lengths = np.zeros(N, dtype=np.int32)
t0 = time.time()
bad = 0
for i, rec in enumerate(man["recording"]):
    fp = os.path.join(FEATURE_DIR, rec.replace(".webm", ".npy"))
    try:
        a = np.load(fp)
        T = min(a.shape[0], MAXF)
        arr[i, :T] = a[:T]
        lengths[i] = T
    except Exception as e:
        bad += 1
        print(f"  BAD {rec}: {e!r}", flush=True)
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{N}  ({(i+1)/(time.time()-t0):.0f}/s)", flush=True)

arr.flush()
del arr
os.replace(OUT + ".tmp", OUT)
np.save(LEN_OUT, lengths)

man["pack_idx"] = range(N)
man.to_csv(MANIFEST, index=False)

print(f"DONE in {(time.time()-t0)/60:.1f} min | bad={bad}", flush=True)
print(f"  features: {OUT}", flush=True)
print(f"  lengths : {LEN_OUT}  (min={int(lengths.min())} max={int(lengths.max())})", flush=True)
print(f"  manifest: added pack_idx column ({N} rows)", flush=True)
