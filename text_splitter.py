# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:51:19 2026

@author: fowak

Text Splitter
"""

import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def semantic_chunker(file_path: str):
    print(f"[SYSTEM] Initializing PyPDFLoader for: {file_path}")
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()
    print(f"[SYSTEM] Successfully extracted {len(raw_documents)} pages from PDF.")
    
    # 2. INITIALIZING THE "SMALL AI" (Free Local Embeddings)
    # We use HuggingFace's MiniLM. It is fast, free, and runs entirely on your local CPU.
    print("[SYSTEM] Booting up local HuggingFace embedding model...")
    local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. THE SMART VALVE (Semantic Chunker)
    # We plug our local embedding model into the Semantic Chunker
    print("[SYSTEM] Configuring Semantic Chunker...")
    semantic_splitter = SemanticChunker(
        embeddings=local_embeddings,
        # 'percentile' means it looks for sharp shifts in the mathematical meaning to make cuts
        breakpoint_threshold_type="percentile" 
    )
    
    # 4. EXECUTE THE SPLIT
    print("[SYSTEM] Pushing raw text through the Semantic Chunker. Standby...")
    # We use .split_documents() instead of .split_text() because we used a Document Loader
    smart_chunks = semantic_splitter.split_documents(raw_documents)
    
    print(f"\n[SUCCESS] Document sliced into {len(smart_chunks)} semantic chunks.")
    
    # --- DIAGNOSTIC PRINTOUT ---
    # Let's inspect the first chunk to verify the cut
    print("\n--- SAMPLE CHUNK 1 ---")
    print(smart_chunks[0].page_content)
    print(f"Metadata (Source File): {smart_chunks[0].metadata['source']}")
    print(f"Metadata (Page Number): {smart_chunks[0].metadata['page']}")
    
    return smart_chunks


def JSON_file_ref(smart_chunks: list):
    # 1. Re-package the LangChain objects into standard Python dictionaries
    export_data = []
    for i, chunk in enumerate(smart_chunks):
        export_data.append({
            "chunk_id": i,
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })
    
    # 2. Open the valve and dump the JSON
    with open("manual_chunks_backup.json", "w", encoding="utf-8") as file:
        # 'indent=4' formats it nicely so it isn't just one massive unreadable line of code
        json.dump(export_data, file, indent=4) 
    
    print("[SUCCESS] Saved to manual_chunks_backup.json")
    
    