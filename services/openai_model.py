import os
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

# https://python.langchain.com/v0.2/docs/integrations/chat/openai/
class OpenAIModel():
    """
    This Class is specific to get a model from Langchain
    and train and return the chain to get informations from document
    """
    def __init__(self, documents):
        self.documents = documents

    def train_llm_and_return(self):
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(self.documents, embeddings)

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=512,
            openai_api_key=os.getenv("OPEN_AI_KEY")
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        return chain, db