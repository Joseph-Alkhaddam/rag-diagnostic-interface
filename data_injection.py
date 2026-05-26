# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:57:24 2026

@author: fowak

Data Injection
"""

from text_splitter import semantic_chunker
from vector_embedding import embedding_and_upsert


file_path = r"C:\Users\fowak\Documents\Work\AI Engineering\RAG Pipeline\Injection Data\Car Manuals\2009 Infinit G37x Coupe\Owner's Manual.pdf"


def main():
    if not file_path:
        print("[CRITICAL] No file path provided. Aborting injection.")
        return
        
    smart_chunks = semantic_chunker(file_path)
    embedding_and_upsert(smart_chunks, "2009-infinit-g37x-data")
    
if __name__ == "__main__":
    main()