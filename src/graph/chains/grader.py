from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.graph.llm import llm

# Pydantic model for the output of the grader chain
class GraderDoc(BaseModel):
    is_doc_relevant : bool = Field(description="Whether the document is relevant to the user question. True if relevant, False otherwise")
    relevance_score : int = Field(description="A score(1-10) indicating the relevance of the document to the user question")

# Do not use both explicit JSON output format in the system prompt and  llm.with_structured_output and the prompt output format at the same time. 
# It can lead to conflicts in the output parsing. Since we are using a structured output format, we should avoid including explicit instructions in the prompt about how to format the output as JSON. 
# The model will be guided by the structured output format defined in the code, so we can focus on providing clear instructions about how to evaluate the relevance of the document without worrying about the output formatting.
GRADE_DOC_PROMPT='''You are a helpful assistant for grading the relevance of retrieved documents to a user question.Given a user question and a retrieved document, your task is to determine whether the document is relevant to the question. 
Here is the user question: {question}
Here is the retrieved document: {document}
'''

# Do not use SystemMessage as we are using variables in the prompt. SystemMessage is meant for static instructions that do not change across different invocations of the prompt. Using variables in a SystemMessage can lead to issues with how the prompt is processed and may not work i.e variables will not be injected and sent to the model. 
grade_doc_prompt = ChatPromptTemplate.from_messages(
    [("system", GRADE_DOC_PROMPT)]
)

# Grader chain that takes in the question and document, and outputs whether the document is relevant along with a relevance score
grader_chain = grade_doc_prompt | llm.with_structured_output(GraderDoc)

if __name__ == "__main__":
    print(grader_chain.invoke(
        {"question": "What is the capital of France?",
        "document": "Paris is the capital of France."}))
    print(grader_chain.invoke(
        {"question": "What is the capital of France?",
        "document": "India is Food capital"}))