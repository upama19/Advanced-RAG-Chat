import sys
from pathlib import Path

# --- Fix imports (project root) ---
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
from generation.rag_answer import answer_question
import pickle
import faiss
from embeddings.build_index import build_index_for_pdf

# ---------- PAGE CONFIG (must be first Streamlit call) ----------
st.set_page_config(page_title="Advanced RAG Chat", page_icon="🤖", layout="centered")

# ---------- STYLING ----------
st.markdown(
    """
<style>
.stApp {
    background-color: white;
}

div[data-testid="chat-message-user"] {
    background-color: #f0f0f0;
    color: black;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="chat-message-assistant"] {
    background-color: white;
    color: black;
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 12px;
}

textarea {
    border: 2px solid black !important;
    border-radius: 8px;
}

button {
    background-color: black !important;
    color: white !important;
    border-radius: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- SIDEBAR : PDF UPLOAD ----------
st.sidebar.title("📄 Upload Research Paper")

uploaded_file = st.sidebar.file_uploader("Upload a PDF paper", type=["pdf"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if uploaded_file is not None and "index_ready" not in st.session_state:
    pdf_path = UPLOAD_DIR / uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):
        index, chunks = build_index_for_pdf(pdf_path)

        # Save index and metadata
        faiss.write_index(index, "embeddings/faiss.index")
        with open("embeddings/chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

    st.sidebar.success(f"Processed: {uploaded_file.name}")

    st.session_state.active_document = uploaded_file.name
    st.session_state.document_uploaded = True
    st.session_state.index_ready = True


# ---------- MAIN TITLE ----------
st.markdown(
    "<h1 style='text-align: center;'>📚 Advanced RAG Chat Assistant</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Ask questions grounded in your research papers</p>",
    unsafe_allow_html=True,
)

st.divider()

# ---------- STOP CHAT UNTIL PDF IS UPLOADED ----------
if "document_uploaded" not in st.session_state:
    st.info("⬅️ Please upload a PDF from the sidebar to start chatting.")
    st.stop()

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY CHAT HISTORY ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- CHAT INPUT ----------
user_input = st.chat_input("Ask about the uploaded paper")

if user_input:
    # User message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, _ = answer_question(
                user_input, active_document=st.session_state.active_document
            )
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------- FOOTER ----------
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with RAG • FAISS • MCP Routing • OpenAI</p>",
    unsafe_allow_html=True,
)
