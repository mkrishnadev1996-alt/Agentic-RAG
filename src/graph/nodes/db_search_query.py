from src.graph.llm import llm
from src.graph.state import GraphState
from src.graph.chains.search_db import db_search_chain

def db_search_query (state : GraphState):
    '''
    Function to generate search queries using the LLM
    Args:
        state: The current state of the graph
    Returns:
        The updated state with the search queries
    '''
    if state.get("question"):
        response = db_search_chain.invoke({"question": state.get("question")})
        return {
            "search_queries": response.retrieve_query
        }
