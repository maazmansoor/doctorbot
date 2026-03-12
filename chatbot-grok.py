import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not OPENAI_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ Missing API keys. Please set OPENAI_API_KEY and GROQ_API_KEY in your .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with your Doctor",
    page_icon="🩺",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* General */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f0f4f8;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Header */
    .header-box {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white;
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(26,115,232,0.3);
    }
    .header-box h1 { margin: 0; font-size: 2rem; }
    .header-box p  { margin: 6px 0 0; opacity: 0.85; font-size: 0.95rem; }

    /* Cards */
    .card {
        background: white;
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        margin-bottom: 20px;
    }

    /* Answer box */
    .answer-box {
        background: #e8f5e9;
        border-left: 5px solid #43a047;
        border-radius: 10px;
        padding: 18px 22px;
        font-size: 1rem;
        line-height: 1.7;
        color: #1b5e20;
    }

    /* Status badges */
    .badge-ready   { background:#d4edda; color:#155724; padding:6px 14px; border-radius:20px; font-size:.85rem; font-weight:600; }
    .badge-missing { background:#fff3cd; color:#856404; padding:6px 14px; border-radius:20px; font-size:.85rem; font-weight:600; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #ffffff; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1a73e8, #0d47a1) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        font-weight: 600 !important;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* Text input */
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1.5px solid #c5d0de !important;
        padding: 10px 14px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1a73e8 !important;
        box-shadow: 0 0 0 3px rgba(26,115,232,0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🩺 Chat with Your Doctor</h1>
    <p>Ask medical questions based on your uploaded PDF documents. Powered by LLaMA 3 + RAG.</p>
</div>
""", unsafe_allow_html=True)

# ── LLM & Prompt ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="Llama3-8b-8192",
    )

llm = get_llm()

prompt_template = ChatPromptTemplate.from_template("""
You are a helpful medical assistant. Answer the question below using ONLY the provided context.
If the answer is not in the context, say: "I couldn't find relevant information in the provided documents."

<context>
{context}
</context>

Question: {input}

Provide a clear, accurate, and helpful answer.
""")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    docs_folder = st.text_input("📁 PDF folder path", value="./doctor")

    chunk_size    = st.slider("Chunk size",    500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0,   500,  200,  step=50)
    max_docs      = st.slider("Max documents", 10,  200,  50,   step=10)

    st.markdown("---")
    st.markdown("### 📊 Status")

    if "vectors" in st.session_state:
        st.markdown('<span class="badge-ready">✅ Vector DB ready</span>', unsafe_allow_html=True)
        st.caption(f"Chunks loaded: {len(st.session_state.get('final_documents', []))}")
    else:
        st.markdown('<span class="badge-missing">⚠️ Not embedded yet</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model:** LLaMA 3 8B  \n**Embeddings:** OpenAI  \n**Vector DB:** FAISS")

# ── Embedding function ────────────────────────────────────────────────────────
def build_vector_store(folder: str, chunk_size: int, chunk_overlap: int, max_docs: int):
    """Load PDFs, split, embed, and store in FAISS."""
    if not os.path.isdir(folder):
        st.error(f"Folder not found: `{folder}`. Create it and add your PDF files.")
        return False

    pdfs = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not pdfs:
        st.error(f"No PDF files found in `{folder}`.")
        return False

    with st.spinner("📄 Loading PDF documents…"):
        loader   = PyPDFDirectoryLoader(folder)
        docs     = loader.load()

    with st.spinner("✂️ Splitting into chunks…"):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(docs[:max_docs])

    with st.spinner("🔢 Generating embeddings (this may take a moment)…"):
        embeddings = OpenAIEmbeddings()
        vectors    = FAISS.from_documents(chunks, embeddings)

    # Persist in session state
    st.session_state.embeddings       = embeddings
    st.session_state.vectors          = vectors
    st.session_state.final_documents  = chunks

    return True

# ── Main layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask a Question")

    question = st.text_input(
        "Your question",
        placeholder="e.g. What are the symptoms of diabetes?",
        label_visibility="collapsed",
    )

    ask_col, clear_col = st.columns([3, 1])
    with ask_col:
        ask_btn = st.button("🔍 Get Answer", use_container_width=True)
    with clear_col:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.pop("last_response", None)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Answer display ────────────────────────────────────────────────────────
    if ask_btn and question:
        if "vectors" not in st.session_state:
            st.warning("⚠️ Please build the Vector DB first using the button on the right.")
        else:
            with st.spinner("🤔 Thinking…"):
                try:
                    doc_chain      = create_stuff_documents_chain(llm, prompt_template)
                    retriever      = st.session_state.vectors.as_retriever(
                                         search_kwargs={"k": 5}
                                     )
                    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

                    t0       = time.perf_counter()
                    response = retrieval_chain.invoke({"input": question})
                    elapsed  = time.perf_counter() - t0

                    st.session_state.last_response = response
                    st.session_state.last_elapsed  = elapsed

                except Exception as e:
                    st.error(f"❌ Error during retrieval: {e}")

    if "last_response" in st.session_state:
        resp    = st.session_state.last_response
        elapsed = st.session_state.get("last_elapsed", 0)

        st.markdown(f"**Answer** *(responded in {elapsed:.2f}s)*")
        st.markdown(
            f'<div class="answer-box">{resp["answer"]}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("📚 Source Chunks Used"):
            for i, doc in enumerate(resp.get("context", []), 1):
                src = doc.metadata.get("source", "Unknown")
                pg  = doc.metadata.get("page", "?")
                st.markdown(f"**Chunk {i}** — `{os.path.basename(src)}` · page {pg}")
                st.text(doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else ""))
                st.markdown("---")

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗄️ Vector Database")
    st.caption("Embed your PDF documents before asking questions.")

    if st.button("⚡ Build / Rebuild Vector DB", use_container_width=True):
        success = build_vector_store(docs_folder, chunk_size, chunk_overlap, max_docs)
        if success:
            st.success(f"✅ Vector DB ready! {len(st.session_state.final_documents)} chunks indexed.")
            st.balloons()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Info card ─────────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📖 How to Use")
    st.markdown("""
1. Place your PDF files in the folder specified in the sidebar (default: `./doctor`)
2. Click **Build / Rebuild Vector DB** to index them
3. Type your medical question and click **Get Answer**
4. Expand *Source Chunks Used* to see which document passages were referenced
""")
    st.markdown("</div>", unsafe_allow_html=True)
