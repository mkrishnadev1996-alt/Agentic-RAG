from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List , Literal
from src.graph.llm import llm_fast

class RouterDestination(BaseModel):
    '''
    Class for defining the router destination
    Attributes:
        destination: The destination for the router. It can be "web_search" to search the web or "db_search" to search the vector database
    '''
    destination: Literal["do_web_search", "db_search"] = Field(description="The destination for the router. It can be 'web_search' to search the web or 'db_search' to search the vector database")


router_prompt = '''You are a router that routes the user question to the appropriate destination based on the question. The possible destinations are "web_search" and "db_search". 
The vector database is used for searching structured data related to AI , Agents, Prompt Engineering, Security related to AI,Adversarial Attacks on LLMs. The web is used for general information retrieval.
If the question is more suitable for web search, route it to "do_web_search". If the question is more suitable for Vector database search, route it to "db_search". Always choose the most appropriate destination based on the user question.
        Question: {question} '''

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", router_prompt)
    ]
    )
router_chain = prompt_template | llm_fast.with_structured_output(RouterDestination)

if __name__ == "__main__":
    # Example usage
    response = router_chain.invoke({"question": "What is Pizza?"})
    print(response.destination)
        