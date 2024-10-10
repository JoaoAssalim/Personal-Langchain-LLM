import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from services.web_searcher import WebSearcher

load_dotenv()


# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
class GroqModel:
    def __init__(self):
        self.llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.0,
            max_retries=3,
            api_key=os.getenv("GROQ_API_KEY")
        )
    
    def answer_quest(self, query, context):
        if not context:
            web_searcher = WebSearcher(results=3)
            context = web_searcher.get_web_information(query=query)
            
        prompt = PromptTemplate(
            template="Based on the following context: {context}.\nAnswer the user query: {query}",
            input_variables=["context", "query"],
        )
        
        chain = prompt | self.llm
        
        response = chain.invoke({"context": context, "query": query})
        return response.content