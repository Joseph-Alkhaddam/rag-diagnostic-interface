# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:36:59 2026

@author: fowak

RAG backend
"""

from openai import OpenAI
from pinecone import Pinecone

index_name = ""

def run_rag_pipeline(user_query, openai_api_key, pinecone_api_key, index_name: str):
    
    # Initialize the manifolds
    ai_client = OpenAI(api_key=openai_api_key)
    pc_client = Pinecone(api_key=pinecone_api_key)
    index = pc_client.Index(index_name)
    
    # 1. Embed the query (Extracting the actual math array!)
    embedding_response = ai_client.embeddings.create(
        input=user_query,
        model="text-embedding-3-small"
    )
    query_vector = embedding_response.data[0].embedding
    
    # 2. Search Pinecone for context (Widened to top 3 results)
    vector_results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    
    print("\n[SYSTEM DIAGNOSTIC] Raw Pinecone Response:")
    print(vector_results)
    print("----------------------------------------\n")
    
    # 3. Compile the context from all 3 retrieved chunks
    compiled_context = ""
    for matched_r in vector_results.matches:
        metadata = matched_r.metadata
        # We extract the text and the page number for citations
        compiled_context += f"\n- Page {metadata.get('page', 'Unknown')}: {metadata.get('text', '')}"
    
    # 4. Construct the System Prompt with the compiled context
    system_rules = f"""
    You are a guide, helper, and search assistant for workers referring to the Vancouver Engineering Design Manual.
    Your job is to help users find information and implement it ONLY using the context provided below.
    Always cite the page number when providing a fact.
    If the answer is not contained in the context below, respond EXACTLY with: "I'm not sure at this time, let me patch you to a human assistant."
    
    CONTEXT PROVIDED: 
    {compiled_context}
    """
    
    # 5. Generate the final answer (Fixed the model name typo)
    chat_response = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_rules},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0
    )
    
    # Extract the final string (Fixed the extraction path)
    final_answer = chat_response.choices[0].message.content
    
    # Return ONLY the final string so Streamlit can print it cleanly
    return final_answer
