"""
Verifies the header-stripping hypothesis — NOT part of the main app.
Confirms that "واكب" lives only in a chunk's metadata (header), never
in its embedded body text.

Run it with:
    .\venv\Scripts\python.exe src\verify_header_stripping.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_chunks_from_marker import build_chunks

chunks = build_chunks()

print("Searching for a chunk whose HEADER (not body text) contains 'واكب'...\n")

for i, c in enumerate(chunks):
    headers_text = " ".join(c["headers"].values())
    if "واكب" in headers_text:
        print(f"Chunk index {i}")
        print(f"Headers: {c['headers']}")
        print(f"Is 'واكب' in the body text? {'واكب' in c['text']}")
        print(f"Body text sent to embedding:\n{c['text']}")
        print(f"Images: {c['images']}")
