"""
Image inspection script v2 - NOT part of the main app.
Adds aspect ratio analysis on top of width/height, since width alone
misclassifies wide-but-flat banner images (e.g. a thin logo strip).
 
Run it with:
    .\venv\Scripts\python.exe src\inspect_images.py
"""
 
from pathlib import Path
from PIL import Image
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "marker_output" / "images"
 
# Tunable thresholds based on what we observed together
MIN_WIDTH = 300          # below this, almost certainly a small logo/icon
MAX_FLAT_RATIO = 2.5     # width/height above this = "flat banner", likely low-value
 
results = []
for path in sorted(IMAGES_DIR.glob("*")):
    try:
        with Image.open(path) as img:
            w, h = img.size
    except Exception:
        continue
    ratio = w / h if h else 0
    results.append((path.name, w, h, ratio))
 
# Classify using both rules together
likely_useful = []
likely_logo_or_banner = []
 
for name, w, h, ratio in results:
    if w < MIN_WIDTH or ratio > MAX_FLAT_RATIO:
        likely_logo_or_banner.append((name, w, h, ratio))
    else:
        likely_useful.append((name, w, h, ratio))
 
print(f"Total images: {len(results)}")
print(f"Likely logo/banner (width<{MIN_WIDTH} OR ratio>{MAX_FLAT_RATIO}): {len(likely_logo_or_banner)}")
print(f"Likely useful screenshots: {len(likely_useful)}")
 
print("\n" + "=" * 70)
print("SAMPLE OF IMAGES FLAGGED AS LOGO/BANNER (first 15)")
print("=" * 70)
for name, w, h, ratio in likely_logo_or_banner[:15]:
    print(f"{name:35s} {w:5d} x {h:<5d}  ratio={ratio:.2f}")
 
print("\n" + "=" * 70)
print("SAMPLE OF IMAGES FLAGGED AS USEFUL (first 15)")
print("=" * 70)
for name, w, h, ratio in likely_useful[:15]:
    print(f"{name:35s} {w:5d} x {h:<5d}  ratio={ratio:.2f}")
 
# Show the widest "flat" outlier to double check our ratio threshold
flat_sorted = sorted(likely_logo_or_banner, key=lambda x: -x[1])
print("\n" + "=" * 70)
print("WIDEST images flagged as logo/banner (sanity check on ratio rule)")
print("=" * 70)
for name, w, h, ratio in flat_sorted[:5]:
    print(f"{name:35s} {w:5d} x {h:<5d}  ratio={ratio:.2f}")
 