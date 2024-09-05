import os

from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()

# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
class ChatLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def train_llm_and_return(self, document_training):
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(document_training, embeddings)

        # https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/
        # https://www.analyticsvidhya.com/blog/2023/12/implement-huggingface-models-using-langchain/
        llm = HuggingFaceHub(
            repo_id="huggingfaceh4/zephyr-7b-alpha", 
            model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":512},
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_HUB_API_TOKEN"),
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        return chain, db

    def split_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=25,
        )

        document_training = text_splitter.split_documents(text)
        return self.train_llm_and_return(document_training)

    def load_file(self):
        file_extension = os.path.basename(self.file_path).split("/")[-1].split(".")[-1]
        
        if file_extension == "pdf":
            loader = PyPDFLoader(
                file_path=self.file_path,
                extract_images=True,
            )
        elif file_extension == "docx":
            loader = Docx2txtLoader(
                file_path=self.file_path,
            )

        documents = loader.load()
        return self.split_text(documents)
