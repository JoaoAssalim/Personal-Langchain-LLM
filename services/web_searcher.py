import os

from langchain_community.retrievers import TavilySearchAPIRetriever
from dotenv import load_dotenv

load_dotenv()


# https://docs.tavily.com/
class WebSearcher:
    def __init__(self, results):
        self.results = results

    def get_web_information(self, query):
        retriever = TavilySearchAPIRetriever(
            api_key=os.getenv("TAVILY_API_KEY"), k=self.results
        )  # instantiate tavily retriver to search on internet and get `k` responses

        response = retriever.invoke(query)
        response = [
            item.page_content
            for item in response
        ]
        return "\n".join(response)
