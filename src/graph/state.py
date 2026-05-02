from typing import Annotated, TypedDict , List
from langchain_core.documents import Document
from pydantic import Field

class GraphState(TypedDict):
    '''
    Class for defining the Graph state
    Attributes:
        question: The users question
        generation: The generated answer from LLM
        is_web_search: If web search is required
        documents: The list of documents from the Vector db
    '''
    question: Annotated[str, Field(description="The users question")]
    generation: Annotated[str, Field(description="The generated answer from LLM")]
    is_web_search: Annotated[bool, Field(description="If web search is required")]
    documents: Annotated[List[Document], Field(description="The list of documents from the Vector db")]
