"""
Build chunks from Marker's markdown output — NOT the final pipeline yet.
This script only builds and inspects chunks (no embeddings, no Chroma)
so we can verify the chunking + image-linking logic before wiring it
into main.py.
 
Design:
  1. Split the markdown by headers (#, ##, ####) so each chunk stays
     inside one logical section instead of a fixed character count.
  2. Any header-section still too long gets further split by
     RecursiveCharacterTextSplitter (character-based, as before).
  3. For every final chunk, scan its text for image references
     (![](filename.jpeg)) and keep only the ones that still exist on
     disk (we already deleted 181 low-value logos/banners).
  4. The image markdown syntax itself is stripped from the text sent
     to embeddings later — it's noise, not semantic content. Images
     are kept purely as metadata for displaying alongside the answer.
  5. IMPORTANT: MarkdownHeaderTextSplitter strips the header text out
     of page_content and keeps it only in metadata. Body text often
     refers back to it with a pronoun ("this service...") instead of
     repeating the name, so a search for the exact section name (e.g.
     "واكب") can fail to match the chunk at all — confirmed by testing.
     Fix: prepend the header text(s) back into the embedded chunk text
     so the section name is always present in what gets embedded.
 
Run it with:
    .\venv\Scripts\python.exe src\build_chunks_from_marker.py
"""
 
import re
from pathlib import Path
 
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKDOWN_PATH = PROJECT_ROOT / "marker_output" / "extracted_text.md"
IMAGES_DIR = PROJECT_ROOT / "marker_output" / "images"
 
# Matches markdown image syntax: ![](_page_4_Picture_7.jpeg)
IMAGE_PATTERN = re.compile(r"!\[.*?\]\((.*?)\)")
 
# If a header-section is longer than this, split it further by characters
MAX_SECTION_CHARS = 1500
CHUNK_OVERLAP = 200
 
# A real section title is short. If Marker mis-detected a long sentence
# as a "####" header, treat it as body text instead of metadata.
MAX_HEADER_LENGTH = 60
 
 
def clean_headers(headers: dict, body_text: str) -> tuple[dict, str]:
    """Move any suspiciously long 'header' back into the body text —
    it's almost certainly a mis-detected sentence, not a real title."""
    real_headers = {}
    demoted_text = []
    for key, value in headers.items():
        if len(value) <= MAX_HEADER_LENGTH:
            real_headers[key] = value
        else:
            demoted_text.append(value)
    if demoted_text:
        body_text = "\n".join(demoted_text) + "\n" + body_text
    return real_headers, body_text
 
 
def extract_images_and_clean_text(text: str) -> tuple[str, list[str]]:
    """Find image filenames referenced in this text, keep only ones
    that survived our earlier filtering (still exist on disk), and
    strip the image markdown syntax out of the text itself."""
    found = IMAGE_PATTERN.findall(text)
    existing_images = [name for name in found if (IMAGES_DIR / name).exists()]
    cleaned_text = IMAGE_PATTERN.sub("", text).strip()
    # Collapse leftover blank lines created by removing image lines
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text, existing_images
 
 
def build_chunks() -> list[dict]:
    raw_text = MARKDOWN_PATH.read_text(encoding="utf-8")
 
    # Step 1: split by headers, so each chunk respects section boundaries
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("####", "header_4"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_sections = header_splitter.split_text(raw_text)
 
    # Step 2: further split any section that's still too long
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_SECTION_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
    )
 
    final_chunks = []
    for section in header_sections:
        if len(section.page_content) > MAX_SECTION_CHARS:
            sub_docs = char_splitter.create_documents(
                [section.page_content], metadatas=[section.metadata]
            )
        else:
            sub_docs = [section]
 
        for doc in sub_docs:
            real_headers, body_with_demoted = clean_headers(doc.metadata, doc.page_content)
            cleaned_text, images = extract_images_and_clean_text(body_with_demoted)
            if not cleaned_text:
                continue  # skip chunks that were only an image with no real text
 
            # Prepend header(s) to the embedded text so the section name
            # is present even when the body only refers to it by pronoun
            # (e.g. "هذه الخدمة") — this is what "واكب" was missing.
            header_line = " - ".join(real_headers.values())
            text_for_embedding = f"{header_line}\n{cleaned_text}" if header_line else cleaned_text
 
            final_chunks.append(
                {
                    "text": text_for_embedding,
                    "headers": real_headers,
                    "images": images,
                }
            )
 
    return final_chunks
 
 
def main() -> None:
    chunks = build_chunks()
    print(f"Total chunks produced: {len(chunks)}\n")
 
    with_images = [c for c in chunks if c["images"]]
    print(f"Chunks with at least one linked image: {len(with_images)}")
    print(f"Chunks with no image: {len(chunks) - len(with_images)}\n")
 
    print("=" * 70)
    print("SAMPLE: first 5 chunks that HAVE images")
    print("=" * 70)
    for c in with_images[:5]:
        print(f"\nHeaders: {c['headers']}")
        print(f"Images: {c['images']}")
        print(f"Text length: {len(c['text'])} chars")
        print(f"Text preview: {c['text'][:200]}...")
        print("-" * 50)
 
 
if __name__ == "__main__":
    main()
 