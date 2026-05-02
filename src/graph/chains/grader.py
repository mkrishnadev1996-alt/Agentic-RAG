from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.graph.llm import llm

class GraderDoc(BaseModel):
    is_doc_relevant: bool = Field(description="Whether the document is relevant to the user question. True if relevant, False otherwise")
    relevance_score : int = Field(description="A score(1-10) indicating the relevance of the document to the user question")

GRADE_DOC_PROMPT='''You are a helpful assistant for grading the relevance of retrieved documents to a user question.Given a user question and a retrieved document, your task is to determine whether the document is relevant to the question. 
Here is the user question: {question}
Here is the retrieved document: {document}
'''

grade_doc_prompt = ChatPromptTemplate.from_messages(
    [("system", GRADE_DOC_PROMPT)]
)

# Grader chain that takes in the question and document, and outputs whether the document is relevant along with a relevance score
grader_chain = grade_doc_prompt | llm.with_structured_output(GraderDoc)

if __name__ == "__main__":
    print(grader_chain.invoke(
        {"question": "What is the capital of France?",
        "document": "Paris is the capital of France."}))