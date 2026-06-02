# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:57:24 2026

@author: fowak

Data Injection
"""

from text_splitter import extract_and_chunk_pdf
from vector_embedding import embedding_and_upsert, sanitize_document_chunks


file_path = r"C:\Users\fowak\Documents\Work\AI Engineering\RAG Pipeline\Injection Data\Building Manuals\Ontario Building Code 2024.pdf"
index_name = "jablonsky-obc-demo"


def main():
    """Data injection execution loop."""
    if not file_path:
        print("[CRITICAL] No file path provided. Aborting injection.")
        return
        
    smart_chunks = extract_and_chunk_pdf(file_path)
    clean_chunks = sanitize_document_chunks(smart_chunks)
    embedding_and_upsert(clean_chunks, index_name)
    
if __name__ == "__main__":
    main()