# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:51:19 2026
@author: fowak

Semantic Text Splitter & Document Extraction Utility
"""

import os
import json
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# Suppress local HuggingFace symlink warnings for cleaner console output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def extract_and_chunk_pdf(file_path: str) -> List[Any]:
    """
    Extracts text from a PDF file and slices it into semantic chunks based on meaning.

    Utilizes a local HuggingFace embedding model to calculate mathematical shifts 
    in context. When the meaning changes sharply (based on percentile thresholds), 
    the document is cleanly cut into distinct, searchable chunks.

    Parameters
    ----------
    file_path : str
        The relative or absolute file path to the target PDF document.

    Returns
    -------
    List[Any]
        A list of LangChain Document objects containing the chunked text and metadata.
    """
    print(f"[SYSTEM] Initializing PyPDFLoader for target: {file_path}")
    
    try:
        loader = PyPDFLoader(file_path)
        raw_documents = loader.load()
        print(f"[SYSTEM] Successfully extracted {len(raw_documents)} pages from PDF.")
    except Exception as e:
        print(f"[ERROR] Failed to load PDF at {file_path}. Details: {str(e)}")
        return []

    print("[SYSTEM] Booting up local HuggingFace embedding model (all-MiniLM-L6-v2)...")
    try:
        local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[ERROR] Failed to initialize local HuggingFace embeddings. Details: {str(e)}")
        return []
    
    print("[SYSTEM] Configuring Semantic Chunker...")
    semantic_splitter = SemanticChunker(
        embeddings=local_embeddings,
        breakpoint_threshold_type="percentile" 
    )
    
    print("[SYSTEM] Pushing raw text through the Semantic Chunker. Standby...")
    try:
        smart_chunks = semantic_splitter.split_documents(raw_documents)
        print(f"\n[SUCCESS] Document sliced into {len(smart_chunks)} semantic chunks.")
        
        # Diagnostic Printout for the first chunk to ensure metadata integrity
        if smart_chunks:
            print("\n--- SAMPLE CHUNK 1 DIAGNOSTIC ---")
            print(f"Content Preview: {smart_chunks[0].page_content[:150]}...")
            print(f"Metadata (Source): {smart_chunks[0].metadata.get('source', 'Unknown')}")
            print(f"Metadata (Page): {smart_chunks[0].metadata.get('page', 'Unknown')}")
            print("---------------------------------\n")
            
        return smart_chunks
    
    except Exception as e:
        print(f"[ERROR] Semantic slicing failed. Details: {str(e)}")
        return []


def export_chunks_to_json(smart_chunks: List[Any], output_filepath: str = "manual_chunks_backup.json") -> None:
    """
    Serializes LangChain Document objects into a standard JSON file.

    Parameters
    ----------
    smart_chunks : List[Any]
        The list of semantic chunk objects to be backed up.
    output_filepath : str, optional
        The destination file path for the JSON export (default is "manual_chunks_backup.json").

    Returns
    -------
    None
    """
    if not smart_chunks:
        print("[WARNING] Empty chunk list provided. Aborting JSON export.")
        return

    export_data = []
    for i, chunk in enumerate(smart_chunks):
        export_data.append({
            "chunk_id": i,
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })
    
    try:
        with open(output_filepath, "w", encoding="utf-8") as file:
            json.dump(export_data, file, indent=4) 
        print(f"[SUCCESS] Successfully backed up {len(export_data)} chunks to {output_filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to write JSON backup file. Details: {str(e)}")


# --- ENTRY POINT FOR STANDALONE TESTING ---
if __name__ == "__main__":
    print("This module is designed to be imported as an ingestion utility.")
    print("Example usage:")
    print("  chunks = extract_and_chunk_pdf('my_document.pdf')")
    print("  export_chunks_to_json(chunks, 'backup.json')")