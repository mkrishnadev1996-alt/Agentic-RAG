from src.graph.chains import grader
from src.ingestion import retriever


def test_grader_chain_output_format():
    question = "What is Agent memory"
    docs = retriever.invoke(question)
    if docs:
        first_doc = docs[0].page_content
        result = grader.grader_chain.invoke(
            {"question": question,
            "document": first_doc}
        )
        assert isinstance(result.relevance_score, int) and  isinstance(result.is_doc_relevant, bool), "relevance_score should be an integer.is_doc_relevant should be a boolean. "

def test_grader_chain_is_doc_relevant_yes():
    question = "What is Agent memory"
    docs = retriever.invoke(question)
    if docs:
        first_doc = docs[0].page_content
        result = grader.grader_chain.invoke(
            {"question": question,
            "document": first_doc}
        )
        assert result.is_doc_relevant == True, "The document should be relevant to the question"
def test_grader_chain_is_doc_relevant_no():
    question = "What is Pizza?"
    docs = retriever.invoke(question)
    if docs:
        first_doc = docs[0].page_content
        result = grader.grader_chain.invoke(
            {"question": question,
            "document": first_doc}
        )
        assert result.is_doc_relevant == False, "The document should not be relevant to the question"

