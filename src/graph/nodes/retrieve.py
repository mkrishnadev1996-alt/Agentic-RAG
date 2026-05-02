from typing import Dict , Any 
from src.ingestion import retriever
from src.graph.state import GraphState

def retrieve(state: GraphState) -> Dict[str, Any]:
    '''
    Function to retrieve documents from the Vector db
    Args:
        state: The current state of the graph
    Returns:
        A dictionary with the retrieved documents
    '''
    question = state.get('question', '')
    if not question:
        raise ValueError("Question is required for retrieval")
    print(f"=====Retrieving documents for question: ====\n{question}")
    docs = retriever.invoke(question)
    return {
        "documents": docs
    }