"""
يحدد بالضبط أي ترتيب (rank) تطلع فيه الخدمات العامة الأربعة الناقصة
عشان نعرف الرقم الصحيح لـk بالاسترجاع الأول (retriever)، بدل التخمين.

Run it with:
    .\\venv\\Scripts\\python.exe src\\debug_k_needed.py
"""

from main_marker import (
    build_embeddings,
    load_api_keys,
    load_or_create_vectorstore,
    normalize_query,
)

QUESTION = "ايش خدمات توكلنا العامة"

# الخدمات اللي المفروض تظهر (حسب مراجعتنا اليدوية لملف extracted_text.md)
EXPECTED_HEADERS = [
    "تعريف رقم الجوال",
    "رمز توكلنا",
    "خدمات احسان",
    "بوابة الطقس",
    "بوابة الاستبيانات",
    "بوابة التطوع",
]


def main() -> None:
    load_api_keys()
    embeddings = build_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)

    normalized = normalize_query(QUESTION)
    print(f"السؤال: {QUESTION!r} → بعد normalize: {normalized!r}\n")

    # نجيب نطاق واسع جداً (20) عشان نشوف بالضبط وين كل خدمة تقع
    results = vectorstore.similarity_search_with_score(normalized, k=20)

    print(f"{'الترتيب':<8}{'المسافة':<10}{'العنوان':<40}{'مطابق لخدمة متوقعة؟'}")
    print("-" * 90)

    found_ranks = {}
    for i, (doc, score) in enumerate(results, start=1):
        headers = doc.metadata.get("headers", "{}")
        # نتأكد هل هذا الـchunk يطابق أي وحدة من الخدمات المتوقعة
        matched = next((h for h in EXPECTED_HEADERS if h in headers), None)
        marker = f"✅ {matched}" if matched else ""
        print(f"{i:<8}{score:<10.4f}{headers[:38]:<40}{marker}")
        if matched and matched not in found_ranks:
            found_ranks[matched] = i

    print("\n" + "=" * 50)
    print("ملخص: أي ترتيب وصلت كل خدمة متوقعة")
    print("=" * 50)
    for h in EXPECTED_HEADERS:
        rank = found_ranks.get(h, "❌ لم تظهر حتى ضمن أفضل 20!")
        print(f"  {h}: {rank}")

    if found_ranks:
        max_rank = max(found_ranks.values())
        print(f"\n👉 أقصى ترتيب لازم نغطيه بـk = {max_rank}")


if __name__ == "__main__":
    main()
