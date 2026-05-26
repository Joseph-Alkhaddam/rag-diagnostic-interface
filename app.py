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
from rag_backend import run_rag_pipeline, INDEX_CONFIGS # Your custom RAG function

# --- SECURITY HANDLER ---
def get_credentials():
    # 1. Try to get keys from Streamlit Cloud's secure vault
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"], st.secrets["PINECONE_API_KEY"]
    except FileNotFoundError:
        # If the Streamlit secrets file doesn't exist locally, just ignore the error
        pass
    except Exception as e:
        # Catch any other Streamlit-specific secret errors
        print(type(e))
        pass
    
    # 2. If not on the cloud, load the local .env file
    load_dotenv()
    return os.getenv("OPENAI_API_KEY"), os.getenv("PINECONE_API_KEY")

openai_api_key, pinecone_api_key = get_credentials()


@st.cache_data
def get_available_indexes(pc_key):
    try:
        pc = Pinecone(api_key=pc_key)
        # Returns a list of strings (e.g., ['production-manual-data', 'client-b-data'])
        return [index.name for index in pc.list_indexes()]
    except Exception as e:
        print(type(e))
        return ["production-manual-data"] # Fallback if API fails


available_indexes = get_available_indexes(pinecone_api_key)


# --- THE MAGIC LINK ROUTER ---
# 1. Look at the web URL and see if a specific index was passed in
query_params = st.query_params
target_index = query_params.get("index", None)

# 2. Find where that index lives in your dropdown list (default to 0 if not found)
default_dropdown_position = 0
if target_index in available_indexes:
    default_dropdown_position = available_indexes.index(target_index)

# 3. Draw the sidebar and force the pre-selection
with st.sidebar:
    st.header("⚙️ Platform Settings")
    index_name = st.selectbox(
        "Select Knowledge Base:", 
        available_indexes,
        index=default_dropdown_position  # <--- THE MAGIC HAPPENS HERE
    )
    st.caption(f"Currently querying: **{index_name}**")

#1. Fetch the specific configuration for whatever index the user selected in the sidebar
current_config = INDEX_CONFIGS.get(index_name, INDEX_CONFIGS["default"])
index_title = INDEX_CONFIGS.get("index_title", "")
index_subheader = INDEX_CONFIGS.get("index_subheader", "")
questions_list = current_config.get("magic_questions", [])

st.title("Working helper")
st.subheader("Ask anything about the relevant documents")

# 1. The RAM Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Initialize a memory slot for button clicks
if "button_query" not in st.session_state:
    st.session_state.button_query = None

# 2. The Screen Refresh
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# st.error(f"X-Ray 1 - Exact Index Name: '{index_name}'")
# st.error(f"X-Ray 2 - Questions Loaded: {questions_list}")

# 3. Dynamically generate the buttons ONLY if questions exist
if questions_list and len(st.session_state.messages) == 0:
    st.markdown("💡 **Try one of the examples below:**")
    
    # Create the exact number of columns needed
    cols = st.columns(len(questions_list))
    
    # Loop through the questions and place each one in its own column
    for index, question in enumerate(questions_list):
        with cols[index]:
            if st.button(question, key=f"magic_btn_{index}"):
                st.session_state.button_query = question

# 4. The standard text input
user_input = st.chat_input("Or type your own specific question here...")

# 5. THE LOGIC GATE: Use the typed input, or the button input if clicked
final_query = user_input or st.session_state.button_query

# 6. Fire the engine
if final_query:
    st.session_state.button_query = None
    
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.write(final_query)
        
    # Fetch AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching company documents..."):
            
            response = run_rag_pipeline(final_query, openai_api_key, pinecone_api_key, index_name)
            
            st.write(response)
            # Save AI response to history
            st.session_state.messages.append({"role": "assistant", "content": response})

    
    
# """
# ***** 
# Use: (streamlit run app.py) 
#     when running the app in command prompt.

# Use: (ssh -o ServerAliveInterval=60 -R 80:127.0.0.1:8501 localhost.run)
#     when running the temporary link in command prompt. 
# *****
# """












