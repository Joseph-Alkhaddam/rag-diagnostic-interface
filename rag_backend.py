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
    "default": {
        "system_prompt": """You are a helpful AI assistant. Answer questions based on the provided context.
        Answer the user's question using the provided context. If the exact answer is missing, 
        summarize what the context DOES say about the topic to try and help them. 
        Always cite page numbers."
        """,
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "top_k": 3
    },
    "alexs-university-psych-notes": {
        "system_prompt": """
        You are a helpful study assistant who is educated and trained in behavioral psychology.
        Answer the user's question using the provided context. If the exact answer is missing, 
        summarize what the context DOES say about the topic to try and help them. 
        Always cite page numbers.
        """,
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "top_k": 5,
        "index_title": "Alex's University Psych Notes Study Helper",
        "index_subheader": "Ask any relevant questions about Alex's study notes from his Psych Major",
        "magic_questions": 
            ["What are Nichmachean Ethics?",
             "What refinement of X-ray technology developed that allowed for structural neuroimaging, and when?",
             "Is intelligence considered one single ability or multiple abilities combined?"
             ]
    },
    "2023-kia-forte-data": {
        "system_prompt": """
        You are a helpful technical assistant who is well versed on the 2023 KIA Forte's Owner's manual, Infotainment
        Reference Guide, and the Warranty Information.
        Answer the user's question using the provided context. If the exact answer is missing, 
        summarize what the context DOES say about the topic to try and help them. 
        Always cite page numbers.
        """,
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "top_k": 5,
        "index_title": "2023 KIA Forte Personal Assistane",
        "index_subheader": """Ask any questions relevant to the Owner's Manual, the Infotainment Quick Reference Guide, and 
        the Warranty Information.""",
        "magic_questions": 
            ["How often should I service my breaks during harsh weather conditions?",
             "Can I have presets for playing certain audio? If so, how do I do that?",
             "If my car is damaged by natural debris, do I qualify for coverage?"
             ]
    },
    "lg-washing-machine-data": {
        "system_prompt": """
        You are a helpful technical assistant who is perfectly versed on the LG WM6998*A Washing Machine Owner's Manual.
        Answer the user's question using the provided context. If the exact answer is missing, 
        summarize what the context DOES say about the topic to try and help them. 
        Always cite page numbers.
        """,
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "top_k": 5,
        "index_title": "LG WM6998*A Washing Machine Peronsal Assitant",
        "index_subheader": """Ask any questions relevant to the Owner's Manual.""",
        "magic_questions": 
            ["How do I clean the rubber gasket for the door?",
             "How do I make the machine use more detergent for a stronger smell?",
             "How can I ensure a strong and clean wash in under 2 hours?"
             ]
    },
    "lg-stove-data": {
        "system_prompt": """
        You are a helpful technical assistant who is perfectly versed on the LG LREN6325 Stove Owner's Manual.
        Answer the user's question using the provided context. If the exact answer is missing, 
        summarize what the context DOES say about the topic to try and help them. 
        Always cite page numbers.
        """,
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "top_k": 5,
        "index_title": "LG LREN6325 Stove Peronsal Assitant",
        "index_subheader": """Ask any questions relevant to the Owner's Manual.""",
        "magic_questions": 
            ["What's the safest way to clean the stove top from oil stains?",
             "Do I need a lot of oil when using the air frying function?",
             "Can you tell me a good way to cook certain foods?"
             ]
    },
    "2009-infinit-g37x-data": {
        "system_prompt": """
        You are a helpful technical assistant who is well versed on the 2009 Infinit G37x Owner's manual, Infotainment
        Reference Guide, and the Warranty Information.
        Answer the user's question using the provided context. If the exact answer is missing, 
        summarize what the context DOES say about the topic to try and help them. 
        Always cite page numbers.
        """,
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "top_k": 5,
        "index_title": "2009 Infinit G37x Personal Assistane",
        "index_subheader": """Ask any questions relevant to the Owner's Manual, the Infotainment Quick Reference Guide, and 
        the Warranty Information.""",
        "magic_questions": 
            ["What is the recommended engine oil viscosity and capacity?",
             "What does it mean if the AWD warning light flashes rapidly on the dashboard?",
             "How do I program the HomeLink Universal Transceiver for my garage door?"
             ]
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
