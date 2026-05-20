# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:15:57 2026

@author: fowak

Vector Embedding
"""

import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv


def embedding_and_upsert(smart_chunks: list):
    load_dotenv()
    ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    production_index = pc_client.Index("shaban-case")
    
    print("[SYSTEM] Running data sanitization filter...")
    # This rebuilds the list, keeping ONLY chunks that have actual letters/numbers after stripping out blank spaces
    clean_chunks = [chunk for chunk in smart_chunks if chunk.page_content.strip()]
    total_chunks = len(clean_chunks)
    
    BATCH_SIZE = 100  # The safety valve for API limits

    print(f"[SYSTEM] Beginning vectorization and upload of {total_chunks} chunks...")
    
    # Loop through the chunks in batches of 100
    for i in range(0, total_chunks, BATCH_SIZE):
        # Slice the current batch from the main list
        batch = clean_chunks[i : i + BATCH_SIZE]
        
        # 1. Extract just the text strings for OpenAI
        texts_to_embed = [chunk.page_content for chunk in batch]
        
        print(f" -> Processing batch {i} to {i + len(batch)}...")
        
        # 2. Call OpenAI to turn this batch of text into vectors
        # (Yes, we are spending real API pennies now!)
        response = ai_client.embeddings.create(
            input=texts_to_embed,
            model="text-embedding-3-small"
            )
        
        # 3. Assemble the Pinecone Payload
        pinecone_payload = []
        for j, chunk in enumerate(batch):
            # Create a unique ID (e.g., "chunk_0", "chunk_1")
            vector_id = f"chunk_{i + j}" 
            # Grab the corresponding math array from OpenAI's response
            vector_math = response.data[j].embedding 
            
            # Package the metadata (Crucial: Pinecone needs the text inside the metadata!)
            vector_metadata = {
                "source": chunk.metadata.get("source", "Unknown"),
                "page": chunk.metadata.get("page", 0),
                "text": chunk.page_content  # We must store the text so we can retrieve it later
            }
            
            # Bind them all together and add to the payload
            pinecone_payload.append({
                "id": vector_id,
                "values": vector_math,
                "metadata": vector_metadata
            })
        
        # 4. Open the valve and inject the batch into Pinecone
        production_index.upsert(vectors=pinecone_payload)
    
    print("\n[SUCCESS] Entire manual successfully pressurized into the Vector Database.")
       
    
    
    