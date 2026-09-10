"""
مقارنة شاملة: Semantic بس مقابل Hybrid (BM25 + Semantic)، على كل الـ25 سؤال
بملف eval_set.json — بنفس منطق التصحيح المستخدم بـrun_eval.py.

كل سؤال يمر بدورة RAG كاملة مرتين (مرة لكل طريقة استرجاع)، بعدها Rerank
ثم LLM — نفس المسار الحقيقي اللي يشوفه المستخدم، لا محاكاة.

تحذير: 25 سؤال × طريقتين = 50 دورة كاملة، استهلاك حقيقي من رصيد Cohere
الشهري (تقريباً 150 استدعاء بين Embed وRerank). شغّله مرة وحدة بوعي.

Run it with:
    .\\venv\\Scripts\\python.exe src\\compare_hybrid_vs_semantic.py
"""

import json
from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq

from main_marker import (
    SYSTEM_PROMPT,
    build_documents,
    build_embeddings,
    load_api_keys,
    load_or_create_vectorstore,
    normalize_query,
    rerank_docs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = PROJECT_ROOT / "eval_set.json"
K = 12
TOP_N = 6


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_chain(retriever):
    """نفس منطق build_rag_chain بالضبط، لكن ياخذ أي retriever جاهز
    (semantic أو hybrid) بدل ما يبنيه بنفسه — عشان نقدر نقارن بينهم."""
    llm = ChatGroq(model="openai/gpt-oss-120b")
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )

    def format_docs(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    answer_chain = (
        {
            "context": lambda p: format_docs(p["docs"]),
            "question": lambda p: p["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        docs=lambda q: rerank_docs(
            normalize_query(q), retriever.invoke(normalize_query(q)), top_n=TOP_N
        ),
        question=RunnablePassthrough(),
    ) | RunnablePassthrough.assign(answer=answer_chain)


def score_case(case: dict, answer: str) -> tuple[bool, list[str]]:
    """نفس منطق التصحيح المستخدم بـrun_eval.py بالضبط."""
    reasons = []

    if case.get("expected_answer_contains_not_found"):
        if "غير موجودة" not in answer:
            reasons.append("توقعنا 'غير موجودة' لكن حاول يجاوب")
        return (len(reasons) == 0, reasons)

    missing = [kw for kw in case.get("expected_keywords", []) if kw not in answer]
    if missing:
        reasons.append(f"مفقود: {missing}")

    present_bad = [kw for kw in case.get("expected_no_keywords", []) if kw in answer]
    if present_bad:
        reasons.append(f"ما كان يفترض يظهر: {present_bad}")

    return (len(reasons) == 0, reasons)


def main() -> None:
    load_api_keys()
    cases = load_eval_set()

    embeddings = build_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": K})

    documents = build_documents()
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = K

    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    semantic_chain = build_chain(semantic_retriever)
    hybrid_chain = build_chain(hybrid_retriever)

    results = []
    print(f"\nتشغيل {len(cases)} سؤال عبر الطريقتين...\n")
    print("=" * 90)

    for case in cases:
        q = case["question"]
        sem_answer = semantic_chain.invoke(q)["answer"]
        sem_pass, sem_reasons = score_case(case, sem_answer)

        hyb_answer = hybrid_chain.invoke(q)["answer"]
        hyb_pass, hyb_reasons = score_case(case, hyb_answer)

        results.append(
            {"case": case, "sem_pass": sem_pass, "hyb_pass": hyb_pass}
        )

        sem_icon = "✅" if sem_pass else "❌"
        hyb_icon = "✅" if hyb_pass else "❌"
        flag = "  ⚠️ اختلاف!" if sem_pass != hyb_pass else ""
        print(f"[{case['id']:>2}] Semantic:{sem_icon}  Hybrid:{hyb_icon}{flag}  — {q}")
        if not sem_pass:
            print(f"       Semantic فشل بسبب: {sem_reasons}")
        if not hyb_pass:
            print(f"       Hybrid فشل بسبب: {hyb_reasons}")

    print("\n" + "=" * 90)
    print("الملخص النهائي")
    print("=" * 90)
    sem_total = sum(1 for r in results if r["sem_pass"])
    hyb_total = sum(1 for r in results if r["hyb_pass"])
    print(f"Semantic: {sem_total}/{len(results)}")
    print(f"Hybrid:   {hyb_total}/{len(results)}")

    diffs = [r for r in results if r["sem_pass"] != r["hyb_pass"]]
    print(f"\nعدد الأسئلة اللي اختلفت فيها النتيجة بين الطريقتين: {len(diffs)}")
    for r in diffs:
        winner = "Hybrid أفضل" if r["hyb_pass"] else "Semantic أفضل"
        print(f"  [{r['case']['id']}] {r['case']['question']} → {winner}")


if __name__ == "__main__":
    main()
