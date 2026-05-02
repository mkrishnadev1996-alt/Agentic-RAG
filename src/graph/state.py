from typing import Annotated, TypedDict , List
from langchain_core.documents import Document
from pydantic import Field

class GraphState(TypedDict):
    '''
    Class for defining the Graph state
    Attributes:
        question: The users question
        retrieve_query: LLM generated query for retrieving relevant documents from the vector db
        generation: The generated answer from LLM
        is_web_search: If web search is required
        search_queries: The list of queries for doing web search
        documents: The list of documents from the Vector db
    '''
    question: Annotated[str, Field(description="The users question")]
    retrieve_query: Annotated[str, Field(description="LLM generated query for retrieving relevant documents from the vector db")]
    generation: Annotated[str, Field(description="The generated answer from LLM")]
    is_web_search: Annotated[bool, Field(description="If web search is required")]
    search_queries: Annotated[List[str], Field(description="The list of queries for doing web search as per user question")]
    documents: Annotated[List[Document], Field(description="The list of documents from the Vector db")]
