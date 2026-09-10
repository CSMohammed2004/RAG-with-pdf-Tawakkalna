"""
تجربة معزولة لـQuery Expansion — بدون أي لمس لـmain_marker.py.

يقارن: الاسترجاع العادي (سؤال واحد) مقابل الاسترجاع الموسّع (سؤال + صياغات
بديلة يولّدها الـLLM)، على نفس السؤال اللي كشف الثغرة ("خدمات توكلنا العامة").

الهدف: نشوف هل الصياغات البديلة فعلاً "تمسك" الخدمات الغايبة (بوابة التطوع،
إحسان، بوابة الاستبيانات) قبل ما نقرر ندمج هذا بالنظام الأساسي.

تحذير: كل تشغيلة تستهلك عدة استدعاءات API (1 Groq لتوليد الصياغات + عدة
Cohere Embed لكل صياغة). شغّله بوعي، مو بشكل متكرر.

Run it with:
    .\\venv\\Scripts\\python.exe src\\debug_query_expansion.py
"""

from langchain_groq import ChatGroq

from main_marker import build_embeddings, load_api_keys, load_or_create_vectorstore, normalize_query

QUESTION = "ايش خدمات توكلنا العامة"
N_VARIATIONS = 3
K_PER_QUERY = 8

# الخدمات اللي عرفنا إنها غايبة حتى من أفضل 20 بالبحث العادي (من debug_k_needed.py)
MISSING_SERVICES = ["خدمات احسان", "بوابة الاستبيانات", "بوابة التطوع"]


def generate_query_variations(question: str, n: int = 3) -> list[str]:
    """يولّد صياغات بديلة للسؤال عبر LLM."""
    llm = ChatGroq(model="openai/gpt-oss-120b")

    prompt = f"""أعد صياغة السؤال التالي بـ{n} طرق مختلفة تحافظ على نفس المعنى
لكن بمفردات وزوايا مختلفة. اكتب كل صياغة بسطر منفصل فقط، بدون ترقيم
أو أي نص إضافي.

السؤال: {question}"""

    response = llm.invoke(prompt)
    variations = [line.strip() for line in response.content.split("\n") if line.strip()]
    return variations[:n]


def check_coverage(docs: list, label: str) -> None:
    """يطبع أي خدمات غايبة موجودة الآن ضمن هذي المجموعة من النتائج."""
    all_headers = " | ".join(doc.metadata.get("headers", "") for doc in docs)
    print(f"\n--- {label} ({len(docs)} نتيجة) ---")
    for service in MISSING_SERVICES:
        found = "✅ موجودة الآن" if service in all_headers else "❌ لسا غايبة"
        print(f"  {service}: {found}")


def main() -> None:
    load_api_keys()
    embeddings = build_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_PER_QUERY})

    normalized = normalize_query(QUESTION)
    print(f"السؤال الأصلي: {QUESTION!r}")

    # --- الخط الأساسي: بحث واحد بس (زي النظام الحالي) ---
    baseline_docs = retriever.invoke(normalized)
    check_coverage(baseline_docs, "الأساس (سؤال واحد بس)")

    # --- التجربة: نولّد صياغات بديلة ---
    print(f"\nنولّد {N_VARIATIONS} صياغات بديلة عبر LLM...")
    variations = generate_query_variations(normalized, N_VARIATIONS)
    for i, v in enumerate(variations, 1):
        print(f"  صياغة {i}: {v}")

    # --- الاسترجاع الموسّع: بحث منفصل لكل صياغة + دمج ---
    all_queries = [normalized] + variations
    seen_content = set()
    merged_docs = []
    for q in all_queries:
        docs = retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                merged_docs.append(doc)

    check_coverage(merged_docs, "بعد الدمج (سؤال أصلي + صياغات بديلة)")

    print(f"\nإجمالي الـchunks الفريدة بعد الدمج: {len(merged_docs)}")


if __name__ == "__main__":
    main()
