# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:42:59 2026

@author: fowak

RAG frontend
"""

import streamlit as st
import os
from dotenv import load_dotenv
from rag_backend import run_rag_pipeline, index_name # Your custom RAG function

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

st.title("Working helper")
st.subheader("Ask anything about the relevant documents")

# 1. The RAM Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. The Screen Refresh
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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












