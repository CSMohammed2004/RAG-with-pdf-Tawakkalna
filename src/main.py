"""
RAG (Retrieval-Augmented Generation) CLI for the Tawakkalna Arabic user guide.

Flow:
  1. Load API keys from .env
  2. Load / reuse a local Chroma vector store of PDF chunks
  3. Retrieve the top matching chunks for each Arabic question
  4. Ask Groq (Llama 3.3) to answer using only that context
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Paths — resolved from this file so the app works no matter where you run it
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "Tawakkalnaar.pdf"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# How many chunks to fetch per question
RETRIEVE_K = 4

# System prompt: answer only from context, and always in Arabic
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
    """Multilingual Cohere embeddings — needed so Arabic queries match Arabic chunks."""
    return CohereEmbeddings(model="embed-multilingual-v3.0")


def ingest_pdf(embeddings: CohereEmbeddings) -> Chroma:
    """Load the PDF, split it, embed the chunks, and persist them to chroma_db/."""
    if not PDF_PATH.exists():
        sys.exit(f"PDF not found: {PDF_PATH}")

    print(f"Loading PDF: {PDF_PATH.name}")
    # PyPDFLoader turns each PDF page into a LangChain Document
    pages = PyPDFLoader(str(PDF_PATH)).load()
    print(f"Loaded {len(pages)} page(s).")

    # RecursiveCharacterTextSplitter tries paragraph/sentence boundaries first,
    # then falls back to smaller units. Overlap keeps context across chunk edges.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunk(s). Embedding and saving to chroma_db/ ...")

    # from_documents embeds every chunk and writes the collection to disk
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print("Vector database ready.")
    return vectorstore


def load_or_create_vectorstore(embeddings: CohereEmbeddings) -> Chroma:
    """Reuse chroma_db/ when it already has data; otherwise embed the PDF once."""
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        if vectorstore._collection.count() > 0:
            print("Found existing chroma_db/ — loading it (skipping re-embedding).")
            return vectorstore

    print("No existing vector data found — indexing the PDF.")
    return ingest_pdf(embeddings)


def build_rag_chain(vectorstore: Chroma):
    """Retriever (top 4 chunks) + Groq chat model, wired as a LangChain chain."""
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

    # LCEL pipeline: question → retrieve → fill prompt → LLM → plain text
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def chat_loop(rag_chain) -> None:
    """Ask Arabic questions in the terminal until the user types exit."""
    print("\nدليل توكلنا — اكتب سؤالك بالعربية. للخروج اكتب: exit\n")

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
        answer = rag_chain.invoke(question)
        print(f"الإجابة:\n{answer}\n")


def main() -> None:
    load_api_keys()
    embeddings = build_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    rag_chain = build_rag_chain(vectorstore)
    chat_loop(rag_chain)


if __name__ == "__main__":
    main()
