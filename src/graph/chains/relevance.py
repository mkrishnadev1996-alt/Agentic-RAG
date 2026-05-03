from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.graph.llm import llm_fast

# Pydantic model for the output of the LLM output relevance to question chain
class RelevanceOutput(BaseModel):
    is_relevant : bool = Field(description="Whether the LLM output is relevant to the question. True if relevant, False otherwise")

# Do not use both explicit JSON output format in the system prompt and  llm.with_structured_output and the prompt output format at the same time. 
# It can lead to conflicts in the output parsing. Since we are using a structured output format, we should avoid including explicit instructions in the prompt about how to format the output as JSON. 
GRADE_DOC_PROMPT='''You are a helpful assistant for grading  of LLM output to validate if it answers the user question. Given a LLM generated output and user question, your task is to determine whether LLM output is relevant to the question. 
Context documents: {question}
LLM output: {generation}
'''

# Do not use SystemMessage as we are using variables in the prompt. SystemMessage is meant for static instructions that do not change across different invocations of the prompt. Using variables in a SystemMessage can lead to issues with how the prompt is processed and may not work i.e variables will not be injected and sent to the model. 
grade_doc_prompt = ChatPromptTemplate.from_messages(
    [("system", GRADE_DOC_PROMPT)]
)

# Grader chain that takes in the question and LLM generation text, and outputs whether the LLM generation is relevant/answer the question 
relevance_chain = grade_doc_prompt | llm_fast.with_structured_output(RelevanceOutput)

if __name__ == "__main__":
    print(relevance_chain.invoke(
        {"generation": "The capital of France is Paris.",
         "question": "What is the capital of France?"}))
