"""
Retrieval diagnostic script — NOT part of the main app.
Investigates WHY retrieval fails for a given question by:
  1. Checking if a target keyword exists in any chunk at all.
  2. Running the actual similarity search against chroma_db_marker
     and printing ALL results with their similarity scores (not just
     the top 4), so we can see how far the correct chunk ranks.

Run it with:
    .\venv\Scripts\python.exe src\debug_retrieval.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_chunks_from_marker import build_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings

CHROMA_DIR = PROJECT_ROOT / "chroma_db_marker"

# --- Question to diagnose ---
QUESTION = "ماهو واكب"
KEYWORD = "واكب"

# --- Step 1: does the keyword exist in the raw chunks at all? ---
print("=" * 70)
print(f"STEP 1: Does '{KEYWORD}' exist in any chunk?")
print("=" * 70)

chunks = build_chunks()
matches = [c for c in chunks if KEYWORD in c["text"]]

if not matches:
    print(f"NOT FOUND in any of the {len(chunks)} chunks. The word never made it")
    print("past chunking — likely lost during Marker extraction or text cleaning.")
else:
    for i, c in enumerate(chunks):
        if KEYWORD in c["text"]:
            print(f"\nFound in chunk index {chunks.index(c)}")
            print(f"Headers: {c['headers']}")
            print(f"Length: {len(c['text'])} chars")
            print(f"Text: {c['text']}")

# --- Step 2: run the actual similarity search and show ALL scores ---
print("\n" + "=" * 70)
print(f"STEP 2: Similarity search for question: \"{QUESTION}\"")
print("=" * 70)

embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
)

# Get similarity scores for ALL stored chunks, not just top 4
total_chunks = vectorstore._collection.count()
results = vectorstore.similarity_search_with_score(QUESTION, k=total_chunks)

print(f"\nTotal chunks in DB: {total_chunks}")
print(f"Showing all results ranked by distance (lower = more similar):\n")

target_rank = None
for rank, (doc, score) in enumerate(results, start=1):
    contains_keyword = KEYWORD in doc.page_content
    marker = " <-- CONTAINS KEYWORD" if contains_keyword else ""
    if contains_keyword and target_rank is None:
        target_rank = rank
    if rank <= 8 or contains_keyword:
        preview = doc.page_content[:60].replace("\n", " ")
        print(f"Rank {rank:3d} | distance={score:.4f} | {preview}...{marker}")

print(f"\n{'='*70}")
if target_rank:
    print(f"RESULT: The chunk with '{KEYWORD}' ranked #{target_rank} out of {total_chunks}.")
    print(f"Since we only retrieve top k=4, this chunk was {'INCLUDED' if target_rank <= 4 else 'MISSED'}.")
else:
    print(f"RESULT: No chunk in the DB contains '{KEYWORD}' — confirms it never made it")
    print("into the vector store at all (see Step 1).")
