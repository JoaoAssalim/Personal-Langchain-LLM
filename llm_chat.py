import os
import tempfile

import streamlit as st
import pandas as pd
from io import StringIO

from services.loader import ChatLoader

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = ""

if "chain" not in st.session_state:
    st.session_state.chain = None

if "db" not in st.session_state:
    st.session_state.db = None

if "ver" not in st.session_state:
    st.session_state.ver = False
    
uploaded_file = st.file_uploader("Choose a file to take your questions", type=["txt", "pdf", "docx"])

if uploaded_file is not None and st.session_state.last_uploaded_file != uploaded_file.name:
    st.session_state.last_uploaded_file = uploaded_file.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    

        with st.spinner('File is loading...'):
            loader = ChatLoader(temp_file_path)
            st.session_state.chain, st.session_state.db = loader.load_file()
        
        st.session_state.ver = True

if st.session_state.ver:
    query = st.text_area(
        "Ask something to chat...",
    )
    
    if st.button("Send"):
        docs = st.session_state.db.similarity_search(query)
        response = st.session_state.chain.run(input_documents=docs, question=query)
        st.write(response.split("Helpful Answer:")[-1])