from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.graph.llm import llm_fast

class WebSearchQuery(BaseModel):
    '''
    Class for defining the web search queries
    Attributes:
        query: The query for doing web search based on user question
    '''
    query: str = Field(description="The query for doing Tavily web search as per user question")


prompt = '''Generate a web search query for the following question.
        The query should be concise and relevant to the question.
        Question: {question} '''

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", prompt)
    ]
    )
search_chain = prompt_template | llm_fast.with_structured_output(WebSearchQuery)

if __name__ == "__main__":
    # Example usage
    response = search_chain.invoke({"question": "What is Langgraph?"})
    print(response.query)
        