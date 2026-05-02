from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
from src.graph.llm import llm_fast

class WebSearchQueries(BaseModel):
    '''
    Class for defining the web search queries
    Attributes:
        queries: The list of queries for doing web search
    '''
    queries: List[str] = Field(description="The list of queries(max 5) for doing Tavily web search as per user question")


prompt = '''Generate a list of search queries for the following question.
        The queries should be concise and relevant to the question. Provide a maximum of 5 queries.
        Question: {question} '''

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", prompt)
    ]
    )
search_chain = prompt_template | llm_fast.with_structured_output(WebSearchQueries)

if __name__ == "__main__":
    # Example usage
    response = search_chain.invoke({"question": "What is Langgraph?"})
    print(response.queries)
        