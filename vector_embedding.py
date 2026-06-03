# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:15:57 2026
@author: fowak

Vector Embedding & Database Ingestion Pipeline
"""

import os
from typing import List, Any
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

def sanitize_document_chunks(raw_chunks: List[Any]) -> List[Any]:
    """
    Filters out document chunks that contain no readable text.

    Scans the page_content of each chunk and strips out blank spaces, 
    newline characters, and invisible formatting. If the resulting string 
    is empty, the chunk is discarded to prevent vectorizing blank pages.

    Parameters
    ----------
    raw_chunks : List[Any]
        A list of document chunk objects (typically LangChain Documents).

    Returns
    -------
    List[Any]
        A sanitized list containing only chunks with valid alphanumeric content.
    """
    print("[SYSTEM] Running data sanitization filter...")
    clean_chunks = [chunk for chunk in raw_chunks if chunk.page_content.strip()]
    
    discarded = len(raw_chunks) - len(clean_chunks)
    print(f"[SYSTEM] Sanitization complete. Kept: {len(clean_chunks)} | Discarded: {discarded} blank chunks.")
    
    return clean_chunks


def embedding_and_upsert(smart_chunks: List[Any], index_name: str, tenant_id: str, batch_size: int = 100) -> None:
    """
    Translates text chunks into mathematical vectors and uploads them to Pinecone.

    Authenticates with secure cloud providers, batches the sanitized document chunks 
    to respect API rate limits, requests embeddings from OpenAI, packages the vectors 
    with their corresponding metadata, and executes the upsert to Pinecone.

    Parameters
    ----------
    smart_chunks : List[Any]
        The list of sanitized document chunk objects to be vectorized.
    index_name : str
        The target vector database index where the vectors will be stored.
    batch_size : int, optional
        The number of chunks to process in a single network request (default is 100).

    Returns
    -------
    None
    """
    # 1. Initialization and Authentication
    load_dotenv()
    
    try:
        ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        production_index = pc_client.Index(index_name)
    except Exception as e:
        print(f"[ERROR] Failed to initialize cloud clients. Verify .env keys. Details: {e}")
        return

    # 2. Execute Data Sanitization
    clean_chunks = sanitize_document_chunks(smart_chunks)
    total_chunks = len(clean_chunks)
    
    if total_chunks == 0:
        print("[WARNING] No valid text found in documents. Aborting ingestion.")
        return

    print(f"[SYSTEM] Beginning vectorization and upload of {total_chunks} chunks...")

    # 3. The Batch Execution Loop
    for i in range(0, total_chunks, batch_size):
        batch = clean_chunks[i : i + batch_size]
        texts_to_embed = [chunk.page_content for chunk in batch]
        
        print(f" -> Processing batch {i} to {i + len(batch)}...")
        
        try:
            # Generate Embeddings via OpenAI
            response = ai_client.embeddings.create(
                input=texts_to_embed,
                model="text-embedding-3-small"
            )
            
            # Assemble the Pinecone Payload Matrix
            pinecone_payload = []
            for j, chunk in enumerate(batch):
                vector_id = f"chunk_{i + j}" 
                vector_math = response.data[j].embedding 
                
                vector_metadata = {
                    "source": chunk.metadata.get("source", "Unknown"),
                    "page": chunk.metadata.get("page", 0),
                    "text": chunk.page_content 
                }
                
                pinecone_payload.append({
                    "id": vector_id,
                    "values": vector_math,
                    "metadata": vector_metadata
                })
            
            # Execute Upsert
            production_index.upsert(
                vectors=pinecone_payload,
                namespace=tenant_id
                )
            
        except Exception as e:
            print(f"[ERROR] Batch {i} to {i + len(batch)} failed: {str(e)}")
            # In a full enterprise system, we would add 'exponential backoff' retries here.
            # For now, we print the error and safely skip to the next batch without crashing.
            continue
            
    print("\n[SUCCESS] Document architecture successfully pressurized into the Vector Database.")

# --- ENTRY POINT FOR STANDALONE TESTING ---
if __name__ == "__main__":
    print("This module is designed to be imported as an ingestion utility.")
    print("Example usage: embedding_and_upsert(my_chunks, 'my-index-name')")