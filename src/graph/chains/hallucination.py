from pydantic import BaseModel, Field, BeforeValidator
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate
from src.graph.llm import llm_fast




# Validator to handle string booleans
def coerce_bool(v):
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)

# Pydantic model for the output of the hallucination chain
class HallucinationOutput(BaseModel):
    is_hallucination: Annotated[bool, BeforeValidator(coerce_bool)]= Field(description="Boolean value whether the LLM output is hallucinated (True)or based on the context documents supplied. True if hallucination, False otherwise")

# Do not use both explicit JSON output format in the system prompt and  llm.with_structured_output and the prompt output format at the same time. 
# It can lead to conflicts in the output parsing. Since we are using a structured output format, we should avoid including explicit instructions in the prompt about how to format the output as JSON. 

GRADE_DOC_PROMPT='''You are a helpful assistant for grading  of LLM output to check for hallucination based on  list of context documents. Given a LLM generated output and context documents list, your task is to determine whether LLM output is hallucinated. 
LLM output: {generation} 

Context documents: {documents}

'''

# Do not use SystemMessage as we are using variables in the prompt. SystemMessage is meant for static instructions that do not change across different invocations of the prompt. Using variables in a SystemMessage can lead to issues with how the prompt is processed and may not work i.e variables will not be injected and sent to the model. 
grade_doc_prompt = ChatPromptTemplate.from_messages(
    [("system", GRADE_DOC_PROMPT)]
)

# Grader chain that takes in the LLM generated text and document, and outputs whether the LLM generation is  hallucinated or grounded based on the context documents supplied.
hallucination_chain = grade_doc_prompt | llm_fast.with_structured_output(HallucinationOutput)

if __name__ == "__main__":
    print(hallucination_chain.invoke(
        {"generation": "The capital of France is Paris.",
         "documents": ["Paris is the capital of France."]}))
