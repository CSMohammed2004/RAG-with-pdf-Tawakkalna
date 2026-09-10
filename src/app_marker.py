"""Streamlit UI for the Marker markdown RAG pipeline defined in main_marker.py.

Parallel to app.py. Reuses main_marker.py's functions — no logic is
duplicated here. Adds an image section: any figures Marker linked to the
retrieved chunks are shown under the answer.

Visual layer only: emerald-green glassmorphism theme, Cairo font for
Arabic. No changes to retrieval, chunking, or answer logic.

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

st.set_page_config(
    page_title="دليل توكلنا (Marker) — اسأل أي سؤال",
    layout="centered",
    page_icon="🌿",
)

# ---------------------------------------------------------------------------
# Theme: deep emerald green + translucent glass panels, RTL, Cairo font.
# Pure CSS injection — no change to any Streamlit widget's behavior.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"], [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif !important;
    }

    /* App background: deep emerald gradient */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 20% 20%, #0a3d2e 0%, #062017 55%, #04140d 100%);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Main content column: subtle max width + spacing */
    .block-container {
        padding-top: 2.5rem;
        max-width: 760px;
    }

    /* Title */
    h1 {
        color: #ecfdf5 !important;
        font-weight: 800 !important;
        text-align: center;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem !important;
    }
    .subtitle {
        text-align: center;
        color: #a7f3d0;
        opacity: 0.75;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Glass panel base style, reused for input row, answer box, image cards */
    .glass-panel {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 1rem;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }

    /* Text input styling */
    [data-testid="stTextInput"] input {
        background: rgba(255, 255, 255, 0.07) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 0.75rem !important;
        color: #ecfdf5 !important;
        padding: 0.75rem 1rem !important;
        font-size: 1.02rem !important;
        direction: rtl;
    }
    [data-testid="stTextInput"] input::placeholder {
        color: rgba(236, 253, 245, 0.4) !important;
    }
    [data-testid="stTextInput"] label {
        color: #a7f3d0 !important;
        font-weight: 600 !important;
    }

    /* Primary button: emerald accent */
    button[kind="primary"] {
        background: linear-gradient(135deg, #059669, #047857) !important;
        border: none !important;
        border-radius: 0.75rem !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 0.55rem 2rem !important;
        box-shadow: 0 4px 16px rgba(5, 150, 105, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.5);
    }

    /* Section subheaders */
    h3 {
        color: #6ee7b7 !important;
        font-weight: 700 !important;
        border-right: 3px solid #059669;
        padding-right: 0.6rem;
        margin-top: 1.75rem !important;
    }

    /* Answer box: glass panel with generous padding */
    .answer-box {
        direction: rtl;
        text-align: right;
        padding: 1.25rem 1.5rem;
        line-height: 1.9;
        white-space: pre-wrap;
        font-size: 1.05rem;
        color: #ecfdf5;
    }

    /* Image cards */
    .img-card {
        border-radius: 0.85rem;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.14);
        background: rgba(255, 255, 255, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .img-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.4);
    }
    [data-testid="stImage"] img {
        border-radius: 0.85rem;
    }
    [data-testid="stImageCaption"] {
        color: #a7f3d0 !important;
        text-align: center !important;
        font-size: 0.8rem !important;
    }

    /* Warning box */
    [data-testid="stAlert"] {
        background: rgba(251, 191, 36, 0.1) !important;
        border: 1px solid rgba(251, 191, 36, 0.3) !important;
        border-radius: 0.75rem !important;
        color: #fef3c7 !important;
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


st.title("🌿 دليل توكلنا")
st.markdown('<div class="subtitle">اسأل أي سؤال عن دليل مستخدم تطبيق توكلنا</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-panel" style="padding: 1.25rem 1.5rem;">', unsafe_allow_html=True)
question = st.text_input("السؤال", placeholder="اكتب سؤالك بالعربية هنا...")
asked = st.button("اسأل", type="primary")
st.markdown("</div>", unsafe_allow_html=True)

if asked:
    if not question.strip():
        st.warning("الرجاء كتابة سؤال أولاً.")
    else:
        rag_chain = get_rag_chain()
        with st.spinner("جاري البحث والإجابة..."):
            answer, images = answer_with_sources(rag_chain, question.strip())

        st.subheader("الإجابة")
        st.markdown(
            f'<div class="glass-panel answer-box">{html.escape(answer)}</div>',
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
                    with col:
                        st.markdown('<div class="img-card">', unsafe_allow_html=True)
                        st.image(
                            str(IMAGES_DIR / name),
                            caption=name,
                            use_container_width=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
