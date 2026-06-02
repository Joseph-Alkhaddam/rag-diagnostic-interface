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

def render_sidebar(available_indexes: list) -> str:
    """
    Renders the sidebar navigation and handles Magic Link URL routing.
    Returns the string name of the currently selected index.
    """
    query_params = st.query_params
    target_index = query_params.get("index", None)

    default_dropdown_position = 0
    if target_index in available_indexes:
        default_dropdown_position = available_indexes.index(target_index)

    with st.sidebar:
        try:
            st.image("big_jablonsky_logo.png", use_column_width=True)
            st.divider() # Adds a clean horizontal line under the logo
        except Exception:
            pass # If the logo is missing, it skips gracefully without crashing the app
            
        st.header("⚙️ Platform Settings")
        index_name = st.selectbox(
            "Select Knowledge Base:", 
            available_indexes,
            index=default_dropdown_position
        )
        st.caption(f"Currently querying: **{index_name}**")
        
    return index_name

def initialize_session_state() -> None:
    """Initializes the Streamlit session state memory matrix."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "button_query" not in st.session_state:
        st.session_state.button_query = None


# --- EXECUTION ENGINE ---

def execute_rag_generation_loop(final_query: str, openai_api_key: str, pinecone_api_key: str, index_name: str) -> None:
    """
    Executes the neural search backend, processes the query, and updates chat history.
    """
    # 1. Nullify the temporary button query state
    st.session_state.button_query = None
    
    # 2. Append and render user message
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.write(final_query)
        
    # 3. Execute backend and render response
    with st.chat_message("assistant"):
        with st.spinner("Searching through relevant documents..."):
            try:
                response = run_rag_pipeline(final_query, openai_api_key, pinecone_api_key, index_name)
                st.write(response)
                
                # Commit successful generation to memory
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Inference Engine Exception: {str(e)}")


# --- MAIN ORCHESTRATOR ---

def main():
    """Main application execution loop."""
    # 1. Authenticate and fetch databases
    openai_api_key, pinecone_api_key = get_credentials()
    available_indexes = get_available_indexes(pinecone_api_key)

    # 2. Render Sidebar & Fetch Configuration
    index_name = render_sidebar(available_indexes)
    current_config = INDEX_CONFIGS.get(index_name, INDEX_CONFIGS["default"])
    
    # 3. Render Headers
    st.title(current_config.get("index_title", "📚 Knowledge Base"))
    st.subheader(current_config.get("index_subheader", "Ask anything about the relevant documents"))

    # 4. Initialize RAM / Memory State
    initialize_session_state()

    # 5. Render Historical Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 6. Render Dynamic Magic Buttons
    questions_list = current_config.get("magic_questions", [])
    if questions_list and len(st.session_state.messages) == 0:
        st.markdown("💡 **Try one of the examples below:**")
        cols = st.columns(len(questions_list))
        
        for index, question in enumerate(questions_list):
            with cols[index]:
                if st.button(question, key=f"magic_btn_{index}"):
                    st.session_state.button_query = question

    # 7. Render Input & Logic Gate
    user_input = st.chat_input("Or type your own specific question here...")
    final_query = user_input or st.session_state.button_query

    # 8. Fire the Engine
    if final_query:
        execute_rag_generation_loop(final_query, openai_api_key, pinecone_api_key, index_name)


# --- ENTRY POINT ---
if __name__ == "__main__":
    main()











