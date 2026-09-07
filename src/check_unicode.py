"""
Unicode diagnostic — NOT part of the main app.
Checks the exact Unicode code points of "واكب" as found in the source
markdown vs. how a normal keyboard would type it, to catch invisible
character mismatches (different alef forms, hidden marks, etc).

Run it with:
    .\venv\Scripts\python.exe src\check_unicode.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKDOWN_PATH = PROJECT_ROOT / "marker_output" / "extracted_text.md"

raw_text = MARKDOWN_PATH.read_text(encoding="utf-8")

# Find the line containing the heading, by locating "اكب" (partial, safer match)
lines = raw_text.split("\n")
for i, line in enumerate(lines):
    if "اكب" in line and "#" in line:
        print(f"Line {i}: {line!r}")
        print("Character-by-character breakdown:")
        for ch in line:
            print(f"  {ch!r}  U+{ord(ch):04X}  name-ish: {ch}")
        print()

# Compare against a normally-typed version
typed_normally = "واكب"
print("=" * 60)
print("Normally-typed 'واكب' breakdown:")
for ch in typed_normally:
    print(f"  {ch!r}  U+{ord(ch):04X}")

print("\n" + "=" * 60)
print(f"Does the file contain the exact string {typed_normally!r}? ", typed_normally in raw_text)
