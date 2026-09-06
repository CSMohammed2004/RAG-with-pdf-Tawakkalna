"""
Comparison script — NOT part of the main app.
Extracts the same pages using pypdf (current) vs pdfplumber (alternative)
so you can compare text quality side by side.

Install pdfplumber first if you don't have it:
    .\\venv\\Scripts\\python.exe -m pip install pdfplumber

Run it with:
    .\\venv\\Scripts\\python.exe src\\compare_extraction.py
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "Tawakkalnaar.pdf"

# Which pages to compare (0-indexed, so page 1 in the app = index 0 here)
PAGES_TO_CHECK = [0, 1, 3, 4]  # corresponds to pages 1, 2, 4, 5 you already saw

print("=" * 70)
print("METHOD 1: pypdf (PyPDFLoader) — what main.py currently uses")
print("=" * 70)

pypdf_pages = PyPDFLoader(str(PDF_PATH)).load()
for i in PAGES_TO_CHECK:
    print(f"\n--- Page {i + 1} (pypdf) ---")
    print(pypdf_pages[i].page_content[:400])
    print("...")

print("\n\n")
print("=" * 70)
print("METHOD 2: pdfplumber — alternative extractor")
print("=" * 70)

try:
    import pdfplumber
except ImportError:
    print("\npdfplumber is not installed. Run:")
    print("  .\\venv\\Scripts\\python.exe -m pip install pdfplumber")
    raise SystemExit(1)

with pdfplumber.open(str(PDF_PATH)) as pdf:
    for i in PAGES_TO_CHECK:
        page = pdf.pages[i]
        text = page.extract_text() or "(no text extracted — likely an image-only page)"
        print(f"\n--- Page {i + 1} (pdfplumber) ---")
        print(text[:400])
        print("...")
