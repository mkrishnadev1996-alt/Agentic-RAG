from langgraph.graph import StateGraph, START,END
from src.graph.state import GraphState
from src.graph.nodes.web_search_queries import web_search_queries
from src.graph.nodes.db_search_query import db_search_query
from src.graph.nodes.retrieved_docs_grader import retrieved_docs_grader
from src.graph.nodes.retrieve import retrieve
from src.graph.nodes.generate import generate

graph_builder = StateGraph(state_schema=GraphState)

# Add Nodes
graph_builder.add_node("web_search_queries", web_search_queries)
graph_builder.add_node("db_search_query", db_search_query)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("retrieved_docs_grader", retrieved_docs_grader)
graph_builder.add_node("generate", generate)

# Add Edges
graph_builder.add_edge(START,"db_search_query")
graph_builder.add_edge("db_search_query","retrieve")
graph_builder.add_edge("retrieve","retrieved_docs_grader")
graph_builder.add_edge("retrieved_docs_grader","generate")
graph_builder.add_edge("generate",END)

graph = graph_builder.compile()

if __name__ == "__main__":
    # Example usage
    initial_state = {
        "question": "What is Langgraph?",
        "is_web_search": False
    }
    final_state = graph.invoke(
        {
            "question":"what is agent memory"
        })
    print(final_state.generation)