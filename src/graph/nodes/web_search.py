from src.graph.state import GraphState
from langchain_tavily import TavilySearch
from langchain_core.documents import Document

tool = TavilySearch(max_results=3)

def web_search (state : GraphState):
    
    print("==Web search==")
    question  = state.get("question", "")
    documents = state.get("documents",[])
    if question and state.get("is_web_search"):
        tavily_results = tool.invoke({
            "query":question
        })["results"]

        content = "\n".join([result["content"] for result in tavily_results])
        doc = Document(page_content=content)
        documents.append(doc)

        return {
            "documents": documents,
            "is_web_search": False
            }
    raise ValueError("Question is required")


if __name__=="__main__":
    state = GraphState(question="what is agent memoty",documents=[Document(page_content="Hello")])
    state2 = GraphState(question="what is agent memoty")
    
    print(web_search(state=state2))
