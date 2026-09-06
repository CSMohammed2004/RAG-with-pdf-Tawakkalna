"""
Marker quick test — NOT part of the main app.
Converts the Tawakkalna PDF with Marker and prints the extracted markdown
for inspection, so we can compare quality against pypdf/pdfplumber.

NOTE: The first run will download Marker's OCR/layout models (Surya).
This can take a few minutes and needs an internet connection once.

Run it with:
    .\venv\Scripts\python.exe src\test_marker.py
"""

from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "Tawakkalnaar.pdf"


def main() -> None:
    print("Loading Marker models (first run downloads them, may take a few minutes)...")
    converter = PdfConverter(artifact_dict=create_model_dict())

    print(f"Converting: {PDF_PATH.name}")
    print("This may take a few minutes on CPU — please wait...\n")

    rendered = converter(str(PDF_PATH))
    text, _, images = text_from_rendered(rendered)

    print("=" * 70)
    print("MARKER OUTPUT (first 3000 characters)")
    print("=" * 70)
    print(text[:3000])
    print("\n" + "=" * 70)
    print(f"Total extracted text length: {len(text)} characters")
    print(f"Number of images extracted: {len(images)}")


if __name__ == "__main__":
    main()
