import streamlit as st
from pypdf import PdfReader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
OLLAMA_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

REJECT_EN = "Sorry, this information is not available in the provided document."
REJECT_TA = "மன்னிக்கவும், இந்த தகவல் கொடுக்கப்பட்ட ஆவணத்தில் இல்லை."

# ---------------- SESSION INIT ----------------
def init_state():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "docs_ready": False,
        "welcome_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---------------- LANGUAGE ----------------
def detect_lang(text):
    for c in text:
        if "\u0B80" <= c <= "\u0BFF":
            return "ta"
    return "en"

# ---------------- PDF PROCESS ----------------
def build_vectorstore(files):
    all_text = ""
    for f in files:
        reader = PdfReader(f)
        for p in reader.pages:
            if p.extract_text():
                all_text += p.extract_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(all_text)
    docs = [Document(page_content=c) for c in chunks]

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return FAISS.from_documents(docs, embeddings)

# ---------------- ANSWER LOGIC ----------------
def answer_question(question):
    lang = detect_lang(question)
    reject = REJECT_TA if lang == "ta" else REJECT_EN

    if not st.session_state.vectorstore:
        return reject

    # ✅ Correct API
    docs = st.session_state.vectorstore.similarity_search(question, k=4)

    if not docs:
        return reject

    context = "\n\n".join(d.page_content for d in docs)

    system_prompt = (
        "You are a polite educational customer-care assistant.\n"
        "Answer ONLY from the context.\n"
        "If answer not present, say rejection message.\n"
        "Be short and clear.\n\n"
        f"Context:\n{context}"
    )

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    response = llm.invoke(
        f"{system_prompt}\n\nQuestion: {question}\nAnswer:"
    ).content.strip()

    if not response or "not available" in response.lower():
        return reject

    return response

# ---------------- UI ----------------
st.set_page_config(page_title="Educational RAG Chatbot", layout="wide")
init_state()

st.title("📘 Educational RAG Chatbot")
st.caption("Friendly support + document-based answers")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("📂 Upload Documents")
    files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)

    if st.button("Process Documents"):
        if files:
            with st.spinner("Processing documents..."):
                st.session_state.vectorstore = build_vectorstore(files)
                st.session_state.docs_ready = True
                st.success("Documents processed successfully")
        else:
            st.warning("Please upload at least one PDF")

# ---------- WELCOME ----------
if not st.session_state.welcome_done:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! 😊 I am your educational support assistant. Upload documents and ask questions."
    })
    st.session_state.welcome_done = True

# ---------- CHAT HISTORY ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- CHAT INPUT (FIXED ORDER) ----------
question = st.chat_input("Type your question here...")

if question:
    # 1️⃣ show user message FIRST
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2️⃣ then answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = answer_question(question)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------- RESET ----------
if st.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.session_state.welcome_done = False
    st.rerun()
