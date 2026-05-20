# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:36:59 2026

@author: fowak

RAG backend
"""

from openai import OpenAI
from pinecone import Pinecone

# --- THE CONFIGURATION REGISTRY ---
# This maps an index name to its specific AI personality and parameters
INDEX_CONFIGS = {
    "production-manual-data": {
        "system_prompt": """
        You are a strict engineering guide for the Vancouver Design Manual. Cite pages.
        Your job is to help users find information and implement it ONLY using the context provided below.
        Always cite the page number when providing a fact.
        "If the user's query is too vague to search the context, do NOT answer. Instead, ask them a clarifying question.
        If the answer is not contained in the context below, respond EXACTLY with: "I'm not sure at this time."
        """,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "top_k": 3
    },
    "hr-policy-data": {
        "system_prompt": """
        You are an empathetic HR assistant. Help employees understand their benefits.
        Your job is to help users find information and implement it ONLY using the context provided below.
        Always cite the page number when providing a fact.
        "If the user's query is too vague to search the context, do NOT answer. Instead, ask them a clarifying question.
        If the answer is not contained in the context below, respond EXACTLY with: "I'm not sure at this time."
        """,
        "model": "gpt-4o",  # Maybe this one needs the smarter model
        "temperature": 0.3,
        "top_k": 5
    },
    # The ultimate fallback if they select an index we haven't mapped yet
    "default": {
        "system_prompt": """You are a helpful AI assistant. Answer questions based on the provided context.
        Your job is to help users find information and implement it ONLY using the context provided below.
        Always cite the page number when providing a fact.
        "If the user's query is too vague to search the context, do NOT answer. Instead, ask them a clarifying question.
        If the answer is not contained in the context below, respond EXACTLY with: "I'm not sure at this time."
        """,
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "top_k": 3
    }
}


def run_rag_pipeline(user_query, openai_api_key, pinecone_api_key, index_name: str):
    
    # Initialize the manifolds
    ai_client = OpenAI(api_key=openai_api_key)
    pc_client = Pinecone(api_key=pinecone_api_key)
    index = pc_client.Index(index_name)
    config = INDEX_CONFIGS.get(index_name, INDEX_CONFIGS["default"])
    
    # 1. Embed the query (Extracting the actual math array!)
    embedding_response = ai_client.embeddings.create(
        input=user_query,
        model="text-embedding-3-small"
    )
    query_vector = embedding_response.data[0].embedding
    
    # 2. Search Pinecone for context (Widened to top 3 results)
    vector_results = index.query(
        vector=query_vector,
        top_k=config["top_k"],
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
    {config["system_prompt"]}
    
    CONTEXT PROVIDED: 
    {compiled_context}
    """
    
    # 5. Generate the final answer (Fixed the model name typo)
    chat_response = ai_client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_rules},
            {"role": "user", "content": user_query}
        ],
        temperature=config["temperature"]
    )
    
    # Extract the final string (Fixed the extraction path)
    final_answer = chat_response.choices[0].message.content
    
    # Return ONLY the final string so Streamlit can print it cleanly
    return final_answer
