import os
import sys
import warnings
from dotenv import load_dotenv
from typing import Any, List, Dict, Union

# --- Core Library Imports ---
from crewai import Agent, Task, Crew, Process
from crewai import BaseLLM
from crewai.tools import BaseTool
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import ConfigDict
import google.generativeai as genai

# --- Configuration ---
load_dotenv()
warnings.filterwarnings("ignore")

# --- Constants and Global Configuration ---
# Robustly check for the Google API Key at startup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("[FATAL ERROR] GOOGLE_API_KEY not found in your environment.")
    print("Please ensure a .env file exists in the same directory as this script and contains your Google AI Studio key.")
    sys.exit(1)

# Document and Model Configuration
PDF_FILE_PATH = "knowledge/Copy of APEX AI company overview.pdf"
QDRANT_COLLECTION_NAME = "rag_collection"
GEMINI_MODEL = "gemini-1.5-flash"


# --- Custom LLM Wrapper for CrewAI ---
class GeminiLLM(BaseLLM):
    """
    A custom CrewAI LLM wrapper for Google Gemini.

    This class bypasses the standard LangChain wrappers to make a direct,
    reliable API call to the Google Generative AI service, ensuring
    compatibility with CrewAI's internal data structures.
    """
    def __init__(self, model: str = GEMINI_MODEL):
        super().__init__(temperature=0.1, model=model)
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model_instance = genai.GenerativeModel(model)

    def call(self, prompt: Union[str, List[Dict[str, str]]], **kwargs: Any) -> str:
        """
        Handles the API call to Google Gemini.

        This method correctly processes the OpenAI-formatted message list that
        CrewAI sends, extracts the content, and passes a simple string to the
        Google GenAI library.

        Args:
            prompt: The prompt from CrewAI, which can be a string or a list of message dicts.

        Returns:
            The text response from the LLM.
        """
        # Handle the list of dictionaries format from CrewAI
        if isinstance(prompt, list):
            prompt_text = prompt[-1]['content']
        else:
            prompt_text = prompt

        try:
            response = self.model_instance.generate_content(prompt_text)
            return response.text
        except Exception as e:
            print(f"[ERROR] GeminiLLM API call failed: {e}")
            return f"An error occurred while communicating with the API: {e}"

    def get_context_window_size(self) -> int:
        """Returns the context window size for the model."""
        return 32768  # A safe, standard context size

    def supports_function_calling(self) -> bool:
        """Declares that this simple wrapper does not support tool/function calling."""
        return False


# --- Custom RAG Tool for Pre-computation ---
class SimpleRAGTool(BaseTool):
    """A CrewAI tool for performing RAG on a local PDF document."""
    name: str = "Knowledge Base Search"
    description: str = "Searches the knowledge base for information related to a query."
    model_config = ConfigDict(extra="allow")

    def __init__(self, pdf_path: str, collection_name: str):
        super().__init__()
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:")
        self._setup_rag(pdf_path)

    def _setup_rag(self, pdf_path: str):
        """Loads, chunks, and indexes the PDF document into a Qdrant collection."""
        print(f"Initializing RAG for document: {os.path.basename(pdf_path)}...")
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        
        self.client.add(collection_name=self.collection_name, documents=chunks, ids=list(range(len(chunks))))
        print("RAG setup complete.")

    def _run(self, query: str) -> str:
        """Performs a semantic search on the indexed document."""
        search_results = self.client.query(
            collection_name=self.collection_name,
            query_text=query,
            limit=3
        )
        result_texts = [hit.document for hit in search_results]
        return "\n\n---\n\n".join(result_texts)


# --- Main Application Logic ---
def main():
    """
    Orchestrates the RAG and CrewAI execution process.
    """
    print("--- Initializing RAG-Powered AI Crew ---")
    
    user_query = input("Please enter your question about the document: ")
    if not user_query:
        print("No query provided. Exiting.")
        sys.exit(0)

    # 1. Pre-computation: Retrieve context from the PDF
    print("Retrieving context from knowledge base...")
    knowledge_base_tool = SimpleRAGTool(pdf_path=PDF_FILE_PATH, collection_name=QDRANT_COLLECTION_NAME)
    context_from_rag = knowledge_base_tool.run(query=user_query)
    print("Context retrieved successfully.")

    # 2. Initialize the custom LLM
    llm = GeminiLLM(model=GEMINI_MODEL)
    print(f"LLM Initialized: {GEMINI_MODEL}")

    # 3. Define the Crew
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

    crew = Crew(
        agents=[synthesizer_agent],
        tasks=[synthesis_task],
        process=Process.sequential,
        verbose=True
    )
    
    # 4. Prepare inputs and execute the crew
    crew_inputs = {
        'query': user_query,
        'rag_context': context_from_rag
    }

    print("\n--- Kicking off Crew Execution ---")
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