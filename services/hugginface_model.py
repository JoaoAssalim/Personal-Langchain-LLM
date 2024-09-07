import os

from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
class HuggingFaceModel:
    def __init__(self, documents):
        self.documents = documents

    def train_llm_and_return(self):
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(self.documents, embeddings)

        # https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/
        # https://www.analyticsvidhya.com/blog/2023/12/implement-huggingface-models-using-langchain/
        llm = HuggingFaceHub(
            repo_id="huggingfaceh4/zephyr-7b-alpha",
            model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512},
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_HUB_API_TOKEN"),
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        return chain, db
