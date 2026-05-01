from typing import Dict , Any 
from ingestion import retriever
from graph.state import GraphState

def retrieve(state: GraphState) -> Dict[str, Any]:
    '''
    Function to retrieve documents from the Vector db
    Args:
        state: The current state of the graph
    Returns:
        A dictionary with the retrieved documents
    '''
    question = state['question']
    if not question:
        raise ValueError("Question is required for retrieval")
    docs = retriever.invoke(question)
    return {
        "documents": docs
    }