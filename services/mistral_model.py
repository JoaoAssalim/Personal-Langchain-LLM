import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate

from services.web_searcher import WebSearcher

load_dotenv()


# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
class MistralModel:
    def __init__(self):
        self.llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            max_retries=2,
        )
    
    def answer_quest(self, query, context):
        if not context:
            print(f"\nSearching on internet: {query}\n")
            web_searcher = WebSearcher(results=3)
            context = web_searcher.get_web_information(query=query)
            
        prompt = PromptTemplate(
            template="Based on the following context: {context}.\nAnswer the user query: {query}",
            input_variables=["context", "query"],
        )
        
        chain = prompt | self.llm
        
        response = chain.invoke({"context": context, "query": query})
        return response.content