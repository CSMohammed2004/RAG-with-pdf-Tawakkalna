"""
Delete low-value images (small logos / flat banners) from
marker_output/images/, keeping only useful screenshots.

Uses the same rule we validated together:
  - width < 300px, OR
  - width/height ratio > 2.5 (flat banner)
Plus one manual exception: the cover logo _page_0_Picture_0.jpeg.

This PERMANENTLY deletes files from marker_output/images/.
A backup of all 370 original images already exists outside this
project, so this is safe to run.

Run it with:
    .\venv\Scripts\python.exe src\delete_low_value_images.py
"""

from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "marker_output" / "images"

MIN_WIDTH = 300
MAX_FLAT_RATIO = 2.5
MANUAL_EXCLUDE = {"_page_0_Picture_0.jpeg"}  # cover logo, confirmed by eye

deleted = []
kept = []

for path in sorted(IMAGES_DIR.glob("*")):
    if path.name in MANUAL_EXCLUDE:
        path.unlink()
        deleted.append(path.name)
        continue

    try:
        with Image.open(path) as img:
            w, h = img.size
    except Exception:
        continue

    ratio = w / h if h else 0
    if w < MIN_WIDTH or ratio > MAX_FLAT_RATIO:
        path.unlink()
        deleted.append(path.name)
    else:
        kept.append(path.name)

print(f"Deleted {len(deleted)} low-value images.")
print(f"Kept {len(kept)} useful images.")
print(f"\nRemaining images in {IMAGES_DIR}: {len(list(IMAGES_DIR.glob('*')))}")
