from dotenv import load_dotenv
import sys
from src.graph.graph import graph

load_dotenv()

def main():
    """Main entry point for Agentic-RAG."""
    # Get question from CLI arg or prompt interactively
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")
    
    if not question.strip():
        print("No question provided.")
        return
    
    print(f"\nProcessing question: {question}\n")
    
    initial_state = {"question": question}
    
    try:
        final_state = graph.invoke(initial_state)
        print("\n=== Final Answer ===")
        print(final_state.get("generation", "No answer generated."))
    except Exception as e:
        print(f"Error during graph execution: {e}")

if __name__ == "__main__":
    main()