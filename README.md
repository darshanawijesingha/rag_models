# 📚 PDF Introduction Extractor & Summarizer

This project automatically extracts the **"Introduction" section** from PDF documents using **semantic similarity search** with `LangChain`, then summarizes the extracted content using a `transformers`-based summarization model (e.g., `google/pegasus-xsum`). 

Both the **raw extracted text** and **summarized output** are saved as `.txt` files in the same directory as the original PDF.

---

## 🚀 Features

- 📄 Loads and splits PDFs into manageable chunks.
- 🤖 Embeds and indexes chunks using **HuggingFace Embeddings** and **FAISS**.
- 🔍 Retrieves content related to the **Introduction** section using similarity search.
- ✨ Summarizes extracted text using a pretrained transformer model.
- 💾 Saves the introduction and summary as `.txt` files beside the source PDF.
- 🗂️ Supports **batch processing** of all PDFs in a given folder.

---

## 🧰 Requirements

Install all dependencies using pip:

```bash
pip install langchain faiss-cpu sentence-transformers transformers pypdf langchain-community pymupdf
