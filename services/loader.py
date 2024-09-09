import os

from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    PyPDFLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)
from langchain.docstore.document import Document
from dotenv import load_dotenv

import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import fitz
from io import BytesIO
import pikepdf
import docx

load_dotenv()


# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
class FileLoader:
    """
    This class is to Load, Split and extract informations from files and return
    a Document list to train the LLM
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def extract_and_process_image_from_pdf(self, pdf_document, documents):
        """
        This functions is to extract images from PDF and add as a Document
        in documents list
        """
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            images = page.get_images(full=True)

            for image_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                image_doc = Document(
                    page_content="Image extracted",
                    metadata={
                        "page": page_number + 1,
                        "image_index": image_index + 1,
                        "source": "image",
                        "image_data": image_bytes,
                    },
                )
                documents.append(image_doc)

        return documents

    def split_text(self, text, metadados, file_type=None):
        """
        This functions is to split texts as Documents and return it
        to train use in the LLM model
        """
        # https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=25,
        )

        document_training = text_splitter.split_documents(text)

        if file_type == "pdf":
            pdf_document = fitz.open(self.file_path)
            document_training = self.extract_and_process_image_from_pdf(
                pdf_document, document_training
            )

        return document_training, metadados

    def load_file(self):
        """
        This functions is separete file and find the loader to split text
        """
        file_extension = os.path.basename(self.file_path).split("/")[-1].split(".")[-1]

        if file_extension == "pdf":
            loader = PyPDFLoader(
                file_path=self.file_path,
                extract_images=True,
            )

            with pikepdf.open(self.file_path) as pdf:
                docinfo = pdf.docinfo
                title = str(docinfo.get("/Title", "No Title"))
                author = str(docinfo.get("/Author", "No Author"))
                summary = str(docinfo.get("/Subject", "No Summary"))
                metadados = {"title": title, "author": author, "summary": summary}

        elif file_extension == "docx":
            loader = Docx2txtLoader(
                file_path=self.file_path,
            )

            doc = docx.Document(self.file_path)
            properties = doc.core_properties
            metadados = {
                "title": properties.title if properties.title else "No Title",
                "author": properties.author if properties.author else "No Author",
                "summary": properties.subject if properties.subject else "No Summary",
            }

        documents = loader.load()
        return self.split_text(documents, metadados, file_extension)
