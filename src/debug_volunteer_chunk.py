"""
Debug script — NOT part of the main app.
Finds which final chunk(s) contain "التطوع" (volunteering) and shows
their full content, to see if the chunk itself looks healthy or if
something went wrong during splitting.

Run it with:
    .\venv\Scripts\python.exe src\debug_volunteer_chunk.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_chunks_from_marker import build_chunks

chunks = build_chunks()

print(f"Total chunks: {len(chunks)}\n")
print("=" * 70)
print("CHUNKS CONTAINING 'التطوع'")
print("=" * 70)

found_any = False
for i, c in enumerate(chunks):
    if "التطوع" in c["text"]:
        found_any = True
        print(f"\n--- Chunk #{i} ---")
        print(f"Headers: {c['headers']}")
        print(f"Images: {c['images']}")
        print(f"Length: {len(c['text'])} chars")
        print("-" * 40)
        print(c["text"])
        print("=" * 70)

if not found_any:
    print("No chunk contains 'التطوع' — it may have been dropped during splitting!")
