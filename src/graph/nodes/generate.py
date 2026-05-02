from src.graph.chains.generate import generate_chain
from src.graph.state import GraphState


def generate(state: GraphState):
    '''
    Function to generate the answer using the LLM
    Args:
        state: The current state of the graph
    Returns:
        The updated state with the generated answer
    '''
    question = state.get('question', '')
    documents = state.get('documents', [])
    if not question:
        raise ValueError("Question is required for generation")
    if not documents:
        raise ValueError("Documents are required for generation")
    print(f"=====Generating answer for question: ====")
    response = generate_chain.invoke({
        "question": question,
        "documents": documents
    })

    return {
        "generation": response

    }