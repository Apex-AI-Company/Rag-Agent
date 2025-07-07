import sys
from crewai import Agent, Task, Crew, Process

# --- Import custom components and configuration ---
from config import PDF_FILE, QDRANT_COLLECTION_NAME, GEMINI_MODEL
from llm import GeminiLLM
from rag_tool import RAGTOOL

def create_crew(llm: GeminiLLM) -> Crew:
    """
    Creates and configures the CrewAI crew with a synthesizer agent and task.
    
    Args:
        llm: An instance of the language model to be used by the agent.
        
    Returns:
        A configured CrewAI Crew instance.
    """
    synthesizer_agent = Agent(
        role="Expert Content Synthesizer",
        goal="Synthesize a clear, concise, and accurate answer to the user's question based *only* on the provided context.",
        backstory="You are an expert in distilling complex information. You receive a user's question and a snippet of relevant text, and your task is to formulate the best possible answer using only that text.",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        tools=[]
    )

    synthesis_task = Task(
        description=(
            "Based ONLY on the following context, provide a clear and concise answer to the user's question.\n"
            "--- CONTEXT ---\n"
            "{rag_context}\n"
            "--- END CONTEXT ---\n\n"
            "USER'S QUESTION: '{query}'"
        ),
        expected_output="A final, well-formatted answer that is directly supported by the provided context.",
        agent=synthesizer_agent,
    )

    return Crew(
        agents=[synthesizer_agent],
        tasks=[synthesis_task],
        process=Process.sequential,
        verbose=True
    )

def main():
    """
    Orchestrates the RAG and CrewAI execution process.
    """
    print("--- Initializing RAG-Powered AI Crew ---")
    
    user_query = input("Please enter your question about the document: ")
    if not user_query.strip():
        print("No query provided. Exiting.")
        sys.exit(0)

    # 1. Pre-computation: Retrieve context from the PDF
    print("\n[Step 1/4] Retrieving context from knowledge base...")
    try:
        knowledge_base_tool = RAGTOOL(
            pdf_path=PDF_FILE, 
            collection_name=QDRANT_COLLECTION_NAME
        )
        context_from_rag = knowledge_base_tool.run(query=user_query)
        print("Context retrieved successfully.")
    except FileNotFoundError:
        print(f"[FATAL ERROR] The document was not found at: {PDF_FILE_PATH}")
        print("Please check the 'PDF_FILE_PATH' in config.py.")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL ERROR] Failed during RAG setup: {e}")
        sys.exit(1)


    # 2. Initialize the custom LLM
    print(f"\n[Step 2/4] Initializing LLM: {GEMINI_MODEL}...")
    llm = GeminiLLM()
    print("LLM Initialized.")

    # 3. Define the Crew
    print("\n[Step 3/4] Creating the analysis crew...")
    crew = create_crew(llm)
    print("Crew created.")
    
    # 4. Prepare inputs and execute the crew
    crew_inputs = {
        'query': user_query,
        'rag_context': context_from_rag
    }

    print("\n[Step 4/4] Kicking off Crew Execution...")
    try:
        result = crew.kickoff(inputs=crew_inputs)
        
        print("\n" + "="*50)
        print("Crew Execution Finished.")
        print("Final Answer:")
        print("="*50)
        print(result)
        print("="*50)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred during crew execution: {e}")

if __name__ == "__main__":
    main()