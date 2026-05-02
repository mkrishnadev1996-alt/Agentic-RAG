from src.graph.llm import llm
from src.graph.state import GraphState
from src.graph.chains.search import search_chain

def search_queries (state : GraphState):
    '''
    Function to generate search queries using the LLM
    Args:
        state: The current state of the graph
    Returns:
        The updated state with the search queries
    '''
    if state.get("is_web_search") and state.get("question"):
        response = search_chain.invoke({"question": state.get("question")})
        return {
            "search_queries": response.queries
        }
