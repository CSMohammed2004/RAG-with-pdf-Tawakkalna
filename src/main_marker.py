"""
RAG CLI for the Tawakkalna Arabic user guide — Marker markdown edition.

This is a PARALLEL pipeline to main.py. It does not touch main.py, app.py,
or the original chroma_db/. Differences from main.py:
  * Source is Marker's markdown output, not the raw PDF via PyPDFLoader.
  * Chunking is header-aware (see build_chunks_from_marker.build_chunks).
  * Each chunk carries the image filenames Marker linked to that section,
    so answers can be shown next to their figures.
  * Vectors are persisted to a separate directory: chroma_db_marker/.

Everything else (embedding model, LLM, k=4 retrieval, system prompt rules)
is kept identical to main.py so old vs new can be compared fairly.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq

# The validated chunking + image-linking logic lives here. Reused as-is.
from build_chunks_from_marker import IMAGES_DIR, build_chunks

# ---------------------------------------------------------------------------
# Paths — resolved from this file so the app works no matter where you run it
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db_marker"  # NEW dir — never the original

# How many chunks to fetch per question (same as main.py)
RETRIEVE_K = 4

# System prompt: answer only from context, and always in Arabic (same as main.py)
SYSTEM_PROMPT = """أنت مساعد يجيب عن أسئلة دليل مستخدم تطبيق توكلنا.

قواعد الإجابة:
- أجب بالعربية فقط.
- اعتمد حصراً على السياق المرفق أدناه. لا تستخدم معلومات خارجية.
- إذا لم تكن الإجابة موجودة في السياق، قل بوضوح أن المعلومة غير موجودة في الدليل.
- كن دقيقاً ومباشراً، واذكر التفاصيل العملية إن وُجدت في السياق.

السياق:
{context}
"""


def load_api_keys() -> None:
    """Read GROQ_API_KEY and COHERE_API_KEY from a .env file into os.environ."""
    load_dotenv(PROJECT_ROOT / ".env")

    missing = [
        name
        for name in ("GROQ_API_KEY", "COHERE_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        sys.exit(
            "Missing API key(s) in .env: "
            + ", ".join(missing)
            + "\nCreate a .env file in the project root with those variables."
        )


def build_embeddings() -> CohereEmbeddings:
    """Multilingual Cohere embeddings — same model as main.py."""
    return CohereEmbeddings(model="embed-multilingual-v3.0")


def build_documents() -> list[Document]:
    """Turn build_chunks() dicts into LangChain Documents.

    Chroma only accepts str/int/float/bool metadata values, so the
    ``headers`` dict and ``images`` list are stored as JSON strings and
    decoded again in :func:`images_from_docs`.
    """
    chunks = build_chunks()
    documents = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "headers": json.dumps(chunk["headers"], ensure_ascii=False),
                    "images": json.dumps(chunk["images"], ensure_ascii=False),
                },
            )
        )
    return documents


def ingest_markdown(embeddings: CohereEmbeddings) -> Chroma:
    """Build chunks from Marker markdown, embed them, persist to chroma_db_marker/."""
    documents = build_documents()
    print(
        f"Built {len(documents)} chunk(s) from Marker markdown. "
        f"Embedding and saving to {CHROMA_DIR.name}/ ..."
    )
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print("Vector database ready.")
    return vectorstore


def load_or_create_vectorstore(embeddings: CohereEmbeddings) -> Chroma:
    """Reuse chroma_db_marker/ when it already has data; otherwise embed once."""
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        if vectorstore._collection.count() > 0:
            print(
                f"Found existing {CHROMA_DIR.name}/ — loading it (skipping re-embedding)."
            )
            return vectorstore

    print("No existing vector data found — indexing the Marker markdown.")
    return ingest_markdown(embeddings)


def images_from_docs(docs) -> list[str]:
    """Collect image filenames from retrieved chunks' metadata.

    De-duplicated, order preserved, and filtered to files that actually
    exist under marker_output/images/.
    """
    ordered: list[str] = []
    for doc in docs:
        raw = doc.metadata.get("images", "[]")
        try:
            names = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            names = []
        for name in names:
            if name not in ordered and (IMAGES_DIR / name).exists():
                ordered.append(name)
    return ordered


def build_rag_chain(vectorstore: Chroma):
    """Retriever (top 4) + Groq chat model.

    Unlike main.py, this chain returns a dict:
        {"question": str, "docs": list[Document], "answer": str}
    so callers can see which chunks were retrieved, not just the answer.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_K})

    llm = ChatGroq(model="openai/gpt-oss-120b")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs) -> str:
        """Join retrieved chunks into one context block for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Given {"docs": [...], "question": "..."} -> answer text
    answer_chain = (
        {
            "context": lambda payload: format_docs(payload["docs"]),
            "question": lambda payload: payload["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # question -> {"docs", "question"} -> + "answer"
    return RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough(),
    ) | RunnablePassthrough.assign(answer=answer_chain)


def answer_with_sources(rag_chain, question: str) -> tuple[str, list[str]]:
    """Run the chain and return (answer_text, existing_image_filenames)."""
    result = rag_chain.invoke(question)
    return result["answer"], images_from_docs(result["docs"])


def chat_loop(rag_chain) -> None:
    """Ask Arabic questions in the terminal until the user types exit."""
    print("\nدليل توكلنا (Marker) — اكتب سؤالك بالعربية. للخروج اكتب: exit\n")

    while True:
        try:
            question = input("السؤال: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nمع السلامة.")
            break

        if not question:
            continue
        if question.lower() == "exit":
            print("مع السلامة.")
            break

        print("جاري البحث والإجابة...\n")
        answer, images = answer_with_sources(rag_chain, question)
        print(f"الإجابة:\n{answer}\n")
        if images:
            print("الصور ذات الصلة: " + ", ".join(images) + "\n")


def main() -> None:
    load_api_keys()
    embeddings = build_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    rag_chain = build_rag_chain(vectorstore)
    chat_loop(rag_chain)


if __name__ == "__main__":
    main()
