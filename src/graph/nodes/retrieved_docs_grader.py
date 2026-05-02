from src.graph.state import GraphState
from src.graph.chains.grader import grader_chain

def retrieved_docs_grader(state: GraphState):
    docs = state.get('documents', [])
    if not docs:
        raise ValueError("Documents are not available for grading")
    question = state.get('question', '')
    filtered_docs = []
    is_web_search = False
    for doc in docs:
        result = grader_chain.invoke({"question": question, "document": doc})
        if  result.is_doc_relevant:
            filtered_docs.append(doc)
        else:
            is_web_search = True
    return{
        "documents" : filtered_docs,
        "is_web_search" : is_web_search
    }

