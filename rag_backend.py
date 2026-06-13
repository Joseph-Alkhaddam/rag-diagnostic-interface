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
        "system_prompt": """
        You are a helpful AI assistant. Answer questions based on the provided context.
        Always cite page numbers.
        """,
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "top_k": 3,
        "index_title": "📚 Knowledge Base",
        "index_subheader": "Ask anything about the relevant documents",
        "magic_questions": []
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
        "index_title": "2023 KIA Forte Personal Assistant",
        "index_subheader": "Ask any questions relevant to the Owner's Manual, the Infotainment Quick Reference Guide, and the Warranty Information.",
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
        "index_subheader": "Ask any questions relevant to the Owner's Manual.",
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
        "index_subheader": "Ask any questions relevant to the Owner's Manual.",
        "magic_questions": 
            ["What's the safest way to clean the stove top from oil stains?",
             "Do I need a lot of oil when using the air frying function?",
             "Can you tell me a good way to cook certain foods?"
             ]
    },
    "jablonsky-obc-demo": {
        "system_prompt": """
        You are a highly precise structural engineering AI assistant specializing in the Ontario Building Code (OBC).
        Answer the user's question using the provided context. If the exact answer is missing, 
        summarize what the context DOES say about the topic to help the engineer. 
        Always cite the exact clause, subsection, and page numbers.
        """,
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "top_k": 5,
        "index_title": "Jablonsky & Partners OBC Assistant",
        "index_subheader": "Query Part 4 (Structural Design) of the Ontario Building Code instantly.",
        "logo": "big_jablonsky_logo.png",
        "magic_questions": [
            "What are the live load reduction factors for multi-story columns and foundations?",
            "What are the specified load combinations for Ultimate Limit States (ULS) under wind and snow?",
            "How is the seismic importance factor (I_E) determined for high-rise residential structures?"
             ]
    }
}


def resolve_namespace(namespace_name: str | None) -> str:
    """
    Normalizes namespace values before sending them to Pinecone.
    Pinecone's visible default namespace is '__default__', not 'default'.
    """
    if not namespace_name or not namespace_name.strip():
        return "__default__"

    namespace = namespace_name.strip()

    if namespace == "default":
        return "__default__"

    return namespace    
    
    
def run_rag_pipeline(
    user_query: str,
    openai_api_key: str,
    pinecone_api_key: str,
    index_name: str = "demo-rags",
    namespace_name: str = "__default__"
) -> str:
    """
    Executes a Retrieval-Augmented Generation (RAG) pipeline for a given query.

    Authenticates with OpenAI and Pinecone, embeds the user query, retrieves relevant 
    contextual chunks from the designated vector database index, and generates a 
    grounded response using the specified LLM configuration.

    Parameters
    ----------
    user_query : str
        The question or command provided by the end user.
    openai_api_key : str
        The secure credential token used for OpenAI embedding and chat generation.
    pinecone_api_key : str
        The secure credential token used to authenticate vector database operations.
    index_name : str
        The target vector index identifier representing the specific data isolation zone.

    Returns
    -------
    str
        The final text response generated by the LLM based on the retrieved context.
    """
    # Standard processing applied for client initialization
    ai_client = OpenAI(api_key=openai_api_key)
    pc_client = Pinecone(api_key=pinecone_api_key)
                         
    resolved_index = index_name.strip() if index_name and index_name.strip() else "demo-rags"
    resolved_namespace = resolve_namespace(namespace_name)
    
    index = pc_client.Index(resolved_index)
    config = INDEX_CONFIGS.get(resolved_index, INDEX_CONFIGS["default"])
    
    # Standard processing applied for vector space translation
    embedding_response = ai_client.embeddings.create(
        input=user_query,
        model="text-embedding-3-small"
    )
    query_vector = embedding_response.data[0].embedding
    
    # Standard processing applied for semantic retrieval
    vector_results = index.query(
        vector=query_vector,
        top_k=config["top_k"],
        namespace=resolved_namespace,
        include_metadata=True
    )
    
    # Standard processing applied for context assembly
    compiled_context = ""
    for matched_r in vector_results.matches:
        metadata = matched_r.metadata
        compiled_context += f"\n- Page {metadata.get('page', 'Unknown')}: {metadata.get('text', '')}"
    
    system_rules = f"""
    {config["system_prompt"]}
    
    CONTEXT PROVIDED: 
    {compiled_context}
    """
    
    # Final inference leap mapping context to query intent
    chat_response = ai_client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system_rules},
            {"role": "user", "content": user_query}
        ],
        temperature=config["temperature"]
    )
    
    return chat_response.choices[0].message.content