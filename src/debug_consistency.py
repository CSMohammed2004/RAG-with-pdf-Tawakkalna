"""
تشخيص: هل عدم الثبات بمرحلة الاسترجاع (retriever) أو بقرار الـLLM؟
 
[1] الاسترجاع الخام - يمرّ الآن على normalize_query بالضبط زي ما يصير
    فعلياً داخل build_rag_chain، عشان [1] يعكس الواقع الحقيقي مو نص خام
    غير معالج.
[2] الجواب الكامل عبر answer_with_sources - نفس المسار اللي يشوفه المستخدم.
 
Run it with:
    .\\venv\\Scripts\\python.exe src\\debug_consistency.py
"""
 
from main_marker import (
    answer_with_sources,
    build_embeddings,
    build_rag_chain,
    load_api_keys,
    load_or_create_vectorstore,
    normalize_query,
)
 
QUESTIONS = [
    "ابي اتطوع",
    "ابي اتطوع؟",
    "ايش خدمات الصحة",
    "ايش خدمات توكلنا",
]
 
RUNS_PER_QUESTION = 3
 
 
def main() -> None:
    load_api_keys()
    embeddings = build_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    rag_chain = build_rag_chain(vectorstore)
 
    for question in QUESTIONS:
        print("\n" + "=" * 70)
        print(f"السؤال: {question!r}")
        normalized = normalize_query(question)
        print(f"بعد normalize_query: {normalized!r}")
        print("=" * 70)
 
        # --- الجزء الأول: نفس النص اللي فعلياً يروح للـretriever داخل
        # build_rag_chain (بعد normalize_query)، مو النص الخام مباشرة ---
        print(f"\n[1] نتائج الاسترجاع على النص المُطبَّع (k=6، بدرجات التشابه):")
        results = vectorstore.similarity_search_with_score(normalized, k=6)
        for i, (doc, score) in enumerate(results):
            headers = doc.metadata.get("headers", "{}")
            has_images = doc.metadata.get("images", "[]") != "[]"
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  {i+1}. distance={score:.4f} | صور={has_images} | {headers}")
            print(f"      نص: {preview}...")
 
        # --- الجزء الثاني: نكرر السؤال كامل (retrieval + LLM) عدة مرات ---
        print(f"\n[2] الجواب النهائي عبر {RUNS_PER_QUESTION} محاولات متتالية:")
        for run in range(1, RUNS_PER_QUESTION + 1):
            answer, images = answer_with_sources(rag_chain, question)
            is_not_found = "غير موجودة" in answer
            print(f"\n  محاولة {run}: {'❌ غير موجودة' if is_not_found else '✅ جاوب'}")
            print(f"  الجواب (أول 150 حرف): {answer[:150]}")
            print(f"  عدد الصور: {len(images)}")
 
 
if __name__ == "__main__":
    main()
 