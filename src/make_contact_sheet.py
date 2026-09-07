"""
Contact sheet generator — NOT part of the main app.
Builds one combined image showing samples across the width range,
so we can visually confirm the width<400 = logo hypothesis together.

Run it with:
    .\venv\Scripts\python.exe src\make_contact_sheet.py
"""

from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "marker_output" / "images"
OUTPUT_FILE = PROJECT_ROOT / "contact_sheet.jpg"

THUMB_SIZE = 150  # each thumbnail is 150x150 max
COLS = 6

# Collect (path, width) for every image
items = []
for path in sorted(IMAGES_DIR.glob("*")):
    try:
        with Image.open(path) as img:
            items.append((path, img.width))
    except Exception:
        continue

# Sort by width so we see the full range from smallest to largest
items.sort(key=lambda x: x[1])

# Pick an evenly spaced sample of ~30 images across the whole width range
sample_count = min(30, len(items))
step = max(1, len(items) // sample_count)
sample = items[::step][:sample_count]

rows = (len(sample) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * THUMB_SIZE, rows * (THUMB_SIZE + 20)), "white")
draw = ImageDraw.Draw(sheet)

for i, (path, width) in enumerate(sample):
    row, col = divmod(i, COLS)
    with Image.open(path) as img:
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        x = col * THUMB_SIZE
        y = row * (THUMB_SIZE + 20)
        sheet.paste(img, (x, y))
        draw.text((x + 2, y + THUMB_SIZE + 2), f"w={width}", fill="red")

sheet.save(OUTPUT_FILE, quality=85)
print(f"Saved contact sheet with {len(sample)} sample images to {OUTPUT_FILE}")
print("Width range shown:", sample[0][1], "to", sample[-1][1])
