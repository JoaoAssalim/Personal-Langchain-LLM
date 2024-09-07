import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
class FileLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def split_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=25,
        )

        document_training = text_splitter.split_documents(text)
        return document_training

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
