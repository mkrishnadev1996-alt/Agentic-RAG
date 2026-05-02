from typing import Dict, Any
from src.graph.state import GraphState
from src.graph.chains.grader import grader_chain

def retrieved_docs_grader(state: GraphState) -> Dict[str, Any]:
    docs = state.get('documents', [])
    if not docs:
        raise ValueError("Documents are not available for grading")
    question = state.get('question', '')
    filtered_docs = []
    is_web_search = False

    # Grade each document for relevance to the question
    # If a document is relevant, add it to the filtered_docs list. If not, set is_web_search to True
    for doc in docs:
        result = grader_chain.invoke({"question": question, "document": doc})
        if  result.is_doc_relevant:
            filtered_docs.append(doc)
        else:
            is_web_search = True
            print("==Irrelevant document detected. Setting is_web_search to True.==")
    return {
        "documents" : filtered_docs,
        "is_web_search" : is_web_search
    }

