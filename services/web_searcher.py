import os

from langchain_community.retrievers import TavilySearchAPIRetriever
from dotenv import load_dotenv

load_dotenv()


# https://docs.tavily.com/
# https://python.langchain.com/v0.2/docs/integrations/retrievers/tavily/
class WebSearcher:
    """
    This Class is specific to do web searchs with Tavily
    if the LLM can't response with the document context
    """
    def __init__(self, results):
        self.results = results

    def get_web_information(self, query):
        retriever = TavilySearchAPIRetriever(
            api_key=os.getenv("TAVILY_API_KEY"), k=self.results
        )  # instantiate tavily retriver to search on internet and get `k` responses

        response = retriever.invoke(query)
        response = [
            {"data": item.page_content, "url": item.metadata.get("source", "")}
            for item in response
        ]
        return response
