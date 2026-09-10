"""
Golden Q&A evaluation runner for the Marker RAG pipeline.

Runs every question in eval_set.json against the live rag_chain,
scores each one automatically (keyword match — no LLM-judge yet),
and prints a per-question + summary report.

This does NOT touch main_marker.py's logic — it just calls the same
build_rag_chain()/answer_with_sources() functions already used by the
CLI and Streamlit app, so results reflect exactly what a real user sees.

Run it with:
    .\\venv\\Scripts\\python.exe src\\run_eval.py
"""

import json
import sys
from pathlib import Path

from main_marker import (
    answer_with_sources,
    build_embeddings,
    build_rag_chain,
    load_api_keys,
    load_or_create_vectorstore,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = PROJECT_ROOT / "eval_set.json"


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def score_case(case: dict, answer: str, images: list[str]) -> tuple[bool, list[str]]:
    """Return (passed, list_of_reasons_for_failure)."""
    reasons = []

    # Case type 1: expects a "not found" refusal
    if case.get("expected_answer_contains_not_found"):
        if "غير موجودة" not in answer:
            reasons.append("توقعنا 'غير موجودة' لكن النظام حاول يجاوب")
        if case.get("expected_images_empty") and images:
            reasons.append(f"توقعنا صفر صور لكن رجعت {len(images)} صورة")
        return (len(reasons) == 0, reasons)

    # Case type 2: expects specific keywords present
    missing = [kw for kw in case.get("expected_keywords", []) if kw not in answer]
    if missing:
        reasons.append(f"كلمات متوقعة مفقودة: {missing}")

    # Case type 3: expects certain keywords to NOT appear (no cross-contamination)
    present_but_shouldnt = [
        kw for kw in case.get("expected_no_keywords", []) if kw in answer
    ]
    if present_but_shouldnt:
        reasons.append(f"كلمات ما كان يفترض تظهر: {present_but_shouldnt}")

    return (len(reasons) == 0, reasons)


def main() -> None:
    if not EVAL_SET_PATH.exists():
        sys.exit(f"ملف eval_set.json غير موجود بـ: {EVAL_SET_PATH}")

    cases = load_eval_set()

    load_api_keys()
    embeddings = build_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    rag_chain = build_rag_chain(vectorstore)

    results = []
    print(f"\nتشغيل {len(cases)} سؤال...\n")
    print("=" * 70)

    for case in cases:
        answer, images = answer_with_sources(rag_chain, case["question"])
        passed, reasons = score_case(case, answer, images)
        results.append({"case": case, "passed": passed, "reasons": reasons})

        status = "✅ نجح" if passed else "❌ فشل"
        print(f"\n[{case['id']}] {status} — ({case['category']})")
        print(f"السؤال: {case['question']}")
        if not passed:
            print(f"الجواب الفعلي: {answer[:200]}...")
            print(f"الصور: {images}")
            for r in reasons:
                print(f"  ⚠ {r}")
        print("-" * 70)

    # Summary by category
    print("\n" + "=" * 70)
    print("الملخص حسب الفئة")
    print("=" * 70)
    categories = {}
    for r in results:
        cat = r["case"]["category"]
        categories.setdefault(cat, {"pass": 0, "total": 0})
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["pass"] += 1

    for cat, stats in categories.items():
        print(f"{cat}: {stats['pass']}/{stats['total']}")

    total_pass = sum(1 for r in results if r["passed"])
    print(f"\nالإجمالي: {total_pass}/{len(results)} ({100 * total_pass // len(results)}%)")


if __name__ == "__main__":
    main()
