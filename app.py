import os
import streamlit as st
import pdfplumber

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama

# ================= CONFIG =================
PDF_FOLDER = "pdfs"
VECTOR_STORE_PATH = "vector_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3:latest"

# ============ STRICT SYSTEM PROMPT ============
SYSTEM_PROMPT = """
You are a strict Class 6 English teacher.

RULES:
1. Answer ONLY from the textbook context.
2. No guessing. No assumptions.
3. If answer not clearly found, say exactly:
   "Textbook-la indha kelvikku podhumana thagaval illa."

FORMAT:
English Answer:
Tamil Explanation (simple Tamil):
"""

# ================= LOAD PDFs =================
def load_pdfs():
    full_text = ""
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(PDF_FOLDER, file)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
    return full_text

# =========== VECTOR STORE ====================
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    raw_text = load_pdfs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_text(raw_text)
    store = FAISS.from_texts(chunks, embeddings)
    store.save_local(VECTOR_STORE_PATH)
    return store

# =========== CONTEXT RETRIEVAL ==============
def retrieve_context(question, store):
    docs = store.similarity_search(question, k=6)
    return "\n\n".join(d.page_content for d in docs)

# =========== ANSWER GENERATION ===============
def generate_answer(question, context):
    if len(context.strip()) < 300:
        return "Textbook-la indha kelvikku podhumana thagaval illa."

    llm = Ollama(model=LLM_MODEL)

    prompt = f"""
{SYSTEM_PROMPT}

TEXTBOOK CONTEXT:
{context}

QUESTION:
{question}
"""
    return llm(prompt).strip()

# ================= UI ========================
st.set_page_config(page_title="Class 6 English RAG", layout="wide")

st.title("📘 Class 6 English RAG Chatbot")
st.caption("Advanced RAG | Strict textbook answers | No wrong answers")

question = st.text_input("Ask your question (Tamil + English allowed):")

if question:
    with st.spinner("Checking textbook..."):
        store = get_vectorstore()
        context = retrieve_context(question, store)
        answer = generate_answer(question, context)

    st.success("Answer")

    if "Textbook-la indha kelvikku" in answer:
        st.warning(answer)
    else:
        st.markdown(answer)
