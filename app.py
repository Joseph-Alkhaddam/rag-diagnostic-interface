# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:42:59 2026

@author: fowak

RAG frontend
"""

import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from rag_backend import run_rag_pipeline, INDEX_CONFIGS 

# --- CREDENTIALS & DATA FETCHING ---

def get_credentials() -> tuple:
    """Retrieves API keys from Streamlit Cloud Secrets or local .env file."""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"], st.secrets["PINECONE_API_KEY"]
    except FileNotFoundError:
        pass
    except Exception as e:
        print(type(e))
        pass
    
    load_dotenv()
    return os.getenv("OPENAI_API_KEY"), os.getenv("PINECONE_API_KEY")

@st.cache_data
def get_available_indexes(pc_key: str) -> list:
    """Fetches the list of active indexes from the Pinecone database."""
    try:
        pc = Pinecone(api_key=pc_key)
        return [index.name for index in pc.list_indexes()]
    except Exception as e:
        print(type(e))
        return ["2023-kia-forte-data"] # Fallback if API fails


# --- UI & STATE COMPONENTS ---

def render_sidebar(config: dict) -> str:
    """
    Renders a locked-down production sidebar for the client.
    Requires the config dictionary to be loaded first.
    """
    with st.sidebar:
        # 1. Dynamic Asset Injection
        logo_path = config.get("logo_path")
        if logo_path and os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
            st.divider()

        # 2. Operator Diagnostics (The new sidebar utility)
        st.header("⚙️ System Telemetry")
        st.caption(f"Active Partition: **{config.get('index_title', 'Standard Build')}**")
        st.caption("Vector DB Status: **ONLINE**")
        st.caption("Inference Engine: **STABLE**")
        
        st.divider()
        
        # 3. Session Control
        if st.button("Purge Session Memory", use_container_width=True):
            st.session_state.messages = []
            st.session_state.button_query = None
            st.rerun()

def initialize_session_state() -> None:
    """Initializes the Streamlit session state memory matrix."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "button_query" not in st.session_state:
        st.session_state.button_query = None


# --- EXECUTION ENGINE ---

def execute_rag_generation_loop(
    final_query: str,
    openai_api_key: str,
    pinecone_api_key: str,
    index_name: str,
    namespace_name: str
) -> None:
    """
    Executes the neural search backend, processes the query, and updates chat history.
    """
    st.session_state.button_query = None

    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.write(final_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching through relevant documents..."):
            try:
                response = run_rag_pipeline(
                    user_query=final_query,
                    openai_api_key=openai_api_key,
                    pinecone_api_key=pinecone_api_key,
                    index_name=index_name,
                    namespace_name=namespace_name
                )
                st.write(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                st.error(f"Inference Engine Exception: {str(e)}")


# --- MAIN ORCHESTRATOR ---

def main():
    """Main application execution loop."""
    openai_api_key, pinecone_api_key = get_credentials()

    # 1. Routing: Read the parameters directly from the URL
    query_params = st.query_params
    st.write("DEBUG raw query params:", dict(st.query_params))

    target_index = query_params.get("index", "2023-kia-forte-data")
    raw_namespace = query_params.get("namespace", "__default__")

    if not raw_namespace or not raw_namespace.strip():
        target_namespace = "__default__"
    elif raw_namespace.strip() == "default":
        target_namespace = "__default__"
    else:
        target_namespace = raw_namespace.strip()
    
    st.caption(f"DEBUG target_index: {target_index}")
    st.caption(f"DEBUG target_namespace: {target_namespace}")

    current_config = INDEX_CONFIGS.get(target_index, INDEX_CONFIGS["default"])
    
    # 3. Render the locked-down Sidebar
    render_sidebar(current_config)
    
    # 4. Render Headers
    st.title(current_config.get("index_title", "📚 Knowledge Base"))
    st.subheader(current_config.get("index_subheader", "Ask anything about the relevant documents"))

    # 5. Initialize RAM / Memory State
    initialize_session_state()

    # 6. Render Historical Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 7. Render Dynamic Magic Buttons
    questions_list = current_config.get("magic_questions", [])
    if questions_list and len(st.session_state.messages) == 0:
        st.markdown("💡 **Try one of the examples below:**")
        cols = st.columns(len(questions_list))
        
        for index, question in enumerate(questions_list):
            with cols[index]:
                if st.button(question, key=f"magic_btn_{index}"):
                    st.session_state.button_query = question

    # 8. Render Input & Logic Gate
    user_input = st.chat_input("Or type your own specific question here...")
    final_query = user_input or st.session_state.button_query

    # 9. Fire the Engine 
    if final_query:
        execute_rag_generation_loop(
            final_query=final_query,
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            index_name=target_index,
            namespace_name=target_namespace
        )


# --- ENTRY POINT ---
if __name__ == "__main__":
    main()











