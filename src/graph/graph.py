from langgraph.graph import StateGraph,END
from src.graph.state import GraphState
from src.graph.nodes.web_search import web_search
from src.graph.nodes.db_search_query import db_search_query
from src.graph.nodes.retrieved_docs_grader import retrieved_docs_grader
from src.graph.nodes.retrieve import retrieve
from src.graph.nodes.generate import generate
from src.graph.chains.router import router_chain
from src.graph.chains import hallucination, relevance

def route_destination(state: GraphState)-> str:
    question = state.get("question", "")
    if not question:
        raise ValueError("Question is required for routing")
    response = router_chain.invoke({"question": question})
    print(f"==Routing to {response.destination}==")
    return response.destination # This will be one of ["do_web_search", "db_search"]

def route_after_grader(state: GraphState) -> str:
    
    if state.get("is_web_search",False):
        return "web_search"
    return "generate"

def check_hallucination_and_relevance(state : GraphState) -> str:
    question = state.get("question")
    generation = state.get("generation")
    docs =  state.get("documents")
    print("==Check  Hallucination ==")

    is_hallucination= hallucination.hallucination_chain.invoke(
        {"generation": generation,
         "documents": docs}
        ).is_hallucination
    if not is_hallucination:
        print("==Check  Relevance ==")

        is_relevant = relevance.relevance_chain.invoke(
        {"generation": generation,
         "question": question}).is_relevant
        if is_relevant:
            return "relevant and not hallucinated"
        else:
            return "not relevant"
    else:
        return "hallucinated"


graph_builder = StateGraph(state_schema=GraphState)

# Add Nodes
graph_builder.add_node("db_search_query", db_search_query)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("retrieved_docs_grader", retrieved_docs_grader)
graph_builder.add_node("generate", generate)
graph_builder.add_node("web_search",web_search)

# Add Edges
graph_builder.set_conditional_entry_point(route_destination,path_map={
    "do_web_search":"web_search",
    "db_search": "db_search_query"
})
graph_builder.add_edge("web_search","generate")
graph_builder.add_edge("db_search_query","retrieve")
graph_builder.add_edge("retrieve","retrieved_docs_grader")
graph_builder.add_conditional_edges("retrieved_docs_grader", route_after_grader, path_map={
    "web_search": "web_search",
    "generate": "generate"
})
graph_builder.add_conditional_edges("generate",check_hallucination_and_relevance,path_map={
    "relevant and not hallucinated": END,
    "not relevant":"web_search",
    "hallucinated":"generate"
    })

# Compile graph
graph = graph_builder.compile()
print(graph.get_graph().draw_mermaid_png)

if __name__ == "__main__":
    # Example usage
    # Save diagram
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph_diagram.png", "wb") as f:
        f.write(png_bytes)
    print("Graph diagram saved to graph_diagram.png")
    initial_state = {
        "question": "What is langgraph?"
    }
    final_state = graph.invoke(initial_state)
    print(final_state)