# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:57:24 2026

@author: fowak

Data Injection
"""

from text_splitter import semantic_chunker
from vector_embedding import embedding_and_upsert, sanitize_document_chunks


file_path = r""
index_name = ""


def main():
    """Data injection execution loop."""
    if not file_path:
        print("[CRITICAL] No file path provided. Aborting injection.")
        return
        
    smart_chunks = semantic_chunker(file_path)
    clean_chunks = sanitize_document_chunks(smart_chunks)
    embedding_and_upsert(clean_chunks, index_name)
    
if __name__ == "__main__":
    main()