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


# Add a sleek sidebar for the user to select the "Brain"
with st.sidebar:
    st.header("⚙️ Platform Settings")
    index_name = st.selectbox("Select Knowledge Base:", available_indexes)
    st.caption(f"Currently querying: **{index_name}**")


st.title("Working helper")
st.subheader("Ask anything about the relevant documents")

# 1. The RAM Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. The Screen Refresh
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 1. Fetch the specific configuration for whatever index the user selected in the sidebar
current_config = INDEX_CONFIGS.get(index_name, INDEX_CONFIGS["default"])
questions_list = current_config.get("magic_questions", [])

# 2. Setup a variable to catch a button click
magic_query = None

# 3. Dynamically generate the buttons ONLY if questions exist
if questions_list:
    st.markdown("💡 **Try one of the examples below:**")
    
    # Create the exact number of columns needed
    cols = st.columns(len(questions_list))
    
    # Loop through the questions and place each one in its own column
    for index, question in enumerate(questions_list):
        with cols[index]:
            if st.button(question):
                magic_query = question

# 4. The standard text input
user_input = st.chat_input("Or type your own specific question here...")

# 5. THE LOGIC GATE: Use the typed input, or the button input if clicked
final_query = user_input or magic_query

# 6. Fire the engine
if final_query:
    # 3. The Input Valve (Using the Walrus Operator :=)
    if user_input := st.chat_input("Enter your query here..."):
        
        # 4. Record User Input
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
    
        # 5. Fire the Engine
        with st.spinner("Querying Vector Database..."):
            ai_response = run_rag_pipeline(user_input, openai_api_key, pinecone_api_key, index_name)
    
        # 6. Record AI Output
        with st.chat_message("assistant"):
            st.markdown(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

    
    
# """
# ***** 
# Use: (streamlit run app.py) 
#     when running the app in command prompt.

# Use: (ssh -o ServerAliveInterval=60 -R 80:127.0.0.1:8501 localhost.run)
#     when running the temporary link in command prompt. 
# *****
# """












