import os
import tempfile

import streamlit as st
import pandas as pd
from io import StringIO

from services.loader import FileLoader
from services.hugginface_model import HuggingFaceModel, HuggingFacePipelineModel
from services.openai_model import OpenAIModel
from services.web_searcher import WebSearcher
from services.embed import Embed


if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = ""

if "chain" not in st.session_state:
    st.session_state.chain = None

if "db" not in st.session_state:
    st.session_state.db = None

if "ver" not in st.session_state:
    st.session_state.ver = False

if "loader_documents" not in st.session_state:
    st.session_state.loader_documents = False


uploaded_file = st.file_uploader(
    "Choose a file to take your questions", type=["pdf", "docx", "csv"]
)

if (
    uploaded_file is not None
    and st.session_state.last_uploaded_file != uploaded_file.name
):
    st.session_state.last_uploaded_file = uploaded_file.name

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

        with st.spinner("File is loading..."):
            st.session_state.loader_documents = FileLoader(temp_file_path).load_file()

        st.session_state.ver = True

if st.session_state.ver:
    
    model_type = st.selectbox(
        "Model Engine Type",
        ("HugginFace Pipeline", "HuggingFace Model", "OpenAI"),
    )

    if model_type == "HugginFace Pipeline": # Searching models to improve because this models is really bad actually
        option = st.selectbox(
            "Model Engine",
            ("google/electra-large-discriminator",),
        )
    elif model_type == "HuggingFace Model":
        option = st.selectbox(
            "Model Engine",
            ("huggingfaceh4/zephyr-7b-alpha", ),
        )
    else:
        option = st.selectbox(
            "Model Engine",
            ("gpt-3.5-turbo",),
        )

    query = st.text_area(
        "Ask something to chat...",
    )

    if st.button("Send"):
        if option == "gpt-3.5-turbo":
            st.session_state.chain, st.session_state.db = OpenAIModel(
                st.session_state.loader_documents
            ).train_llm_and_return()
            
            docs = st.session_state.db.similarity_search(query)
            response = st.session_state.chain.run(input_documents=docs, question=query, return_only_outputs=True)
        
        elif option == "huggingfaceh4/zephyr-7b-alpha":
            st.session_state.chain, st.session_state.db = HuggingFaceModel(
                st.session_state.loader_documents, option
            ).train_llm_and_return()
            
            docs = st.session_state.db.similarity_search(query, k=5)
            response = st.session_state.chain.run(input_documents=docs, question=query, return_only_outputs=True)
            
        else:

            st.session_state.chain, st.session_state.db = HuggingFacePipelineModel(
                st.session_state.loader_documents, option
            ).train_llm_and_return()

            docs = st.session_state.db.similarity_search(query, k=5)
            context = " ".join(doc.page_content for doc in docs)
            response = st.session_state.chain(question=query, context=context)["answer"]
            
        
        # if we are using “stuff” as a chain_type, we need to use this type of answer split
        answer = response.split("Helpful Answer:")[-1]
        
        # else, we use this answer splitter for refine type
        # studing something to extract jus the final answer from refine chain
        
        relevance_score = Embed(query, answer).get_score_from_embeds()
        print(answer, relevance_score)

        if answer and relevance_score >= 0.70:
            st.title("Informations found in document.")
            st.write(answer)    
        else:
            st.title("Informations found on web.")
            web_searcher = WebSearcher(3)
            web_response = web_searcher.get_web_information(query)

            for response in web_response:
                st.write(response["data"])
                st.markdown(f"[Link para a informação]({response['url']})")
                st.write("---")
