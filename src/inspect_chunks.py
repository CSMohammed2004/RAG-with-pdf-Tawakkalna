"""
Inspection script — NOT part of the main app.
Loads the PDF, splits it the same way main.py does, and prints the first
few chunks so you can see exactly how the text got divided.

Run it with:
    .\\venv\\Scripts\\python.exe src\\inspect_chunks.py
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "Tawakkalnaar.pdf"

# How many chunks to print
NUM_TO_SHOW = 5

pages = PyPDFLoader(str(PDF_PATH)).load()
print(f"Loaded {len(pages)} page(s).\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunk(s).\n")
print("=" * 70)

for i, chunk in enumerate(chunks[:NUM_TO_SHOW]):
    print(f"\n--- CHUNK #{i} ---")
    print(f"Source page (metadata): {chunk.metadata.get('page', 'unknown')}")
    print(f"Length: {len(chunk.page_content)} characters")
    print("-" * 40)
    print(chunk.page_content)
    print("=" * 70)
