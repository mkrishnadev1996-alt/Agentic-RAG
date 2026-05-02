from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.graph.llm import llm_fast

class DBSearchQueries(BaseModel):
    '''
    Class for defining the vecor DB search queries
    Attributes:
        retrieve_query: The query for doing Vector DB search
    '''
    retrieve_query: str = Field(description="The LLM generated query for doing Vector DB similarity search as per user question")


prompt = '''Generate a search query for the below user question. The generated query should be concise and should be suitable for retrieving relevant documents from a vector database.
        User question: {question} '''


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", prompt)
    ]
    )
db_search_chain = prompt_template | llm_fast.with_structured_output(DBSearchQueries)

if __name__ == "__main__":
    # Example usage
    response = db_search_chain.invoke({"question": " Langgraph use"})
    print(response.retrieve_query)