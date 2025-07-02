
!pip install langchain faiss-cpu sentence-transformers transformers pypdf langchain-community quiet pymupdf

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import re
import os
from langchain.document_loaders import PyPDFLoader
from pathlib import Path

# --- Step 1: Load and Split PDF ---
def extract_chunks_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)
    return chunks

# --- Step 2: Embed and Index Chunks ---
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# --- Step 3: Retrieve "Introduction" Chunks ---
def retrieve_introduction(vectorstore, top_k=5):
    retriever = vectorstore.as_retriever(search_type="similarity", k=top_k)
    results = retriever.get_relevant_documents("Extract the Introduction section")
    intro_text = "\n".join([doc.page_content for doc in results])
    return intro_text

# --- Step 4: Summarize the Introduction ---
def summarize_text(text, model_name="google/pegasus-xsum"):
    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summaries = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
    final_summary = " ".join([s['summary_text'] for s in summaries])
    return final_summary

# --- Run the Full Pipeline ---
def extract_and_summarize_intro(pdf_path):
    chunks = extract_chunks_from_pdf(pdf_path)
    vectorstore = create_vector_store(chunks)
    intro = retrieve_introduction(vectorstore)

    print("\n🧾 --- Extracted Introduction ---\n")
    print(intro)

    summary = summarize_text(intro)
    print("\n📝 --- Summarized Introduction ---\n")
    print(summary)

# Example usage:
pdf_file_path = "/content/file.pdf"  # 🔁 Replace with your PDF path
extract_and_summarize_intro(pdf_file_path)

# --- PDF Loading and Chunking ---
def extract_chunks_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    return chunks

# --- Embedding and Vector Store Creation ---
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# --- Retrieve Introduction Chunks ---
def retrieve_introduction(vectorstore, top_k=5):
    retriever = vectorstore.as_retriever(search_type="similarity", k=top_k)
    results = retriever.get_relevant_documents("Extract the Introduction section")
    intro_text = "\n".join([doc.page_content for doc in results])
    return intro_text

# --- Summarization ---
def summarize_text(text, model_name="google/pegasus-xsum"):
    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summaries = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
    final_summary = " ".join([s['summary_text'] for s in summaries])
    return final_summary

# --- Process Single PDF ---
def process_pdf(pdf_path, output_dir):
    chunks = extract_chunks_from_pdf(pdf_path)
    vectorstore = create_vector_store(chunks)
    intro = retrieve_introduction(vectorstore)
    summary = summarize_text(intro)

    # File name processing
    pdf_name = Path(pdf_path).stem

    # Save raw intro
    intro_file = os.path.join(output_dir, f"{pdf_name}_introduction.txt")
    with open(intro_file, "w", encoding="utf-8") as f:
        f.write(intro)

    # Save summary
    summary_file = os.path.join(output_dir, f"{pdf_name}_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"✅ Processed: {pdf_name}\n🧾 Saved intro: {intro_file}\n📝 Saved summary: {summary_file}\n")

# --- Batch Process Folder of PDFs ---
def process_all_pdfs(folder_path):
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    for pdf_path in pdf_files:
        process_pdf(pdf_path, folder_path)

# --- Set your path ---
# Replace this with your actual folder in Drive
folder_path = "/content"  # 🔁 Change this

process_all_pdfs(folder_path)





