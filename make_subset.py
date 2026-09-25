"""
Copies a random subset of the "140k Real and Fake Faces" dataset
into the data/ folder layout used by fake_face_classifier.py.

Edit SRC to point at the folder that contains train/, valid/ and test/
(after unzipping the Kaggle download it is usually something like
 .../real_vs_fake/real-vs-fake).
"""
import random
import shutil
from pathlib import Path

SRC = Path(r"C:\Users\mrjit\Downloads\real vs fake\real_vs_fake\real-vs-fake")   # <-- change this
DST = Path("data")

# images PER CLASS (real and fake each get this many)
SIZES = {"train": 2000, "valid": 500, "test": 500}

random.seed(42)

for split, n in SIZES.items():
    for cls in ("real", "fake"):
        files = list((SRC / split / cls).glob("*.jpg"))
        chosen = random.sample(files, n)
        out = DST / split / cls
        out.mkdir(parents=True, exist_ok=True)
        for f in chosen:
            shutil.copy(f, out / f.name)
        print(f"{split}/{cls}: copied {len(chosen)} images")

print("Done.")