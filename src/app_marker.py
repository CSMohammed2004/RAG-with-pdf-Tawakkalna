"""Streamlit UI for the Marker markdown RAG pipeline defined in main_marker.py.

Parallel to app.py. Reuses main_marker.py's functions — no logic is
duplicated here. Adds an image section: any figures Marker linked to the
retrieved chunks are shown under the answer.
.\venv\Scripts\python.exe -m streamlit run src\app_marker.py
"""

import html

import streamlit as st

from main_marker import (
    IMAGES_DIR,
    answer_with_sources,
    build_embeddings,
    build_rag_chain,
    load_api_keys,
    load_or_create_vectorstore,
)

st.set_page_config(page_title="دليل توكلنا (Marker) — اسأل أي سؤال", layout="centered")

# Right-to-left layout so Arabic title, inputs, and answers read naturally
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        direction: rtl;
        text-align: right;
    }
    .answer-box {
        direction: rtl;
        text-align: right;
        background-color: rgba(128, 128, 128, 0.15);
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 0.5rem;
        padding: 1rem 1.25rem;
        line-height: 1.8;
        white-space: pre-wrap;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

IMAGES_PER_ROW = 3


@st.cache_resource
def get_vectorstore():
    """Load API keys, embeddings, and Chroma once per Streamlit process."""
    load_api_keys()
    embeddings = build_embeddings()
    return load_or_create_vectorstore(embeddings)


@st.cache_resource
def get_rag_chain():
    """Build the RAG chain once; reuse the cached vector store."""
    vectorstore = get_vectorstore()
    return build_rag_chain(vectorstore)


st.title("دليل توكلنا (Marker) — اسأل أي سؤال")

question = st.text_input("السؤال", placeholder="اكتب سؤالك بالعربية هنا...")
asked = st.button("اسأل", type="primary")

if asked:
    if not question.strip():
        st.warning("الرجاء كتابة سؤال أولاً.")
    else:
        rag_chain = get_rag_chain()
        with st.spinner("جاري البحث والإجابة..."):
            answer, images = answer_with_sources(rag_chain, question.strip())

        st.subheader("الإجابة")
        st.markdown(
            f'<div class="answer-box">{html.escape(answer)}</div>',
            unsafe_allow_html=True,
        )

        # Only render an image section when the retrieved context actually
        # linked figures that exist on disk.
        if images:
            st.subheader("صور ذات صلة")
            for start in range(0, len(images), IMAGES_PER_ROW):
                row = images[start : start + IMAGES_PER_ROW]
                cols = st.columns(IMAGES_PER_ROW)
                for col, name in zip(cols, row):
                    col.image(
                        str(IMAGES_DIR / name),
                        caption=name,
                        use_container_width=True,
                    )
