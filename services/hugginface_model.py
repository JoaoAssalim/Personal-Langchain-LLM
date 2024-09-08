import os

from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import pipeline

from dotenv import load_dotenv


load_dotenv()


# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
class HuggingFaceModel:
    """
    This Class is specific to get a model from Langchain
    and train and return the chain to get informations from document
    """

    def __init__(self, documents, model):
        self.documents = documents
        self.model = model

    def train_llm_and_return(self):
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(self.documents, embeddings)

        # https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/
        # https://www.analyticsvidhya.com/blog/2023/12/implement-huggingface-models-using-langchain/
        llm = HuggingFaceHub(
            repo_id=self.model,
            model_kwargs={"temperature": 1, "max_length": 256, "max_new_tokens": 250},
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_HUB_API_TOKEN"),
            task="text2text-generation",
        )

        chain = load_qa_chain(
            llm, chain_type="stuff"
        )  # We have this four options as “stuff”, “map_reduce”, “map_rerank”, and “refine”. But we just can use stuff and refine because the two other options needs more documents to analyse

        return chain, db


class HuggingFacePipelineModel:
    """
    This Class is specific to get a model from Langchain
    and train and return the chain to get informations from document
    """

    def __init__(self, documents, model):
        self.documents = documents
        self.model = model

    def train_llm_and_return(self):
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(self.documents, embeddings)

        # https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/
        # https://www.analyticsvidhya.com/blog/2023/12/implement-huggingface-models-using-langchain/
        qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.model,
            device=0,
        )

        return qa_pipeline, db
