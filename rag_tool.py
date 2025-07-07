import os 
import sys
from qdrant_client import QdrantClient 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import ConfigDict
from crewai.tools import BaseTool
import fitz  # PyMuPDF

class RAGTOOL(BaseTool):
    """
    A CrewAI tool for managing RAG on a local PDF document.
    """
    name: str = "Knowledge Base Search"
    description: str = "Searches a local PDF document for relevant info"
    model_config = ConfigDict(extra='allow')

    def __init__(self,pdf_path: str,collection_name: str):
        super().__init__()
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:")
        self._setup_rag(pdf_path)

    def _setup_rag(self,pdf_path:str):
        """
        Loads , chunks and indexes the PDF document into Qrant Collection
        """ 
        print(f"Loading PDF doc from {os.path.basename(pdf_path)}...")
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            print("PDF loaded successfully.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
            chunks = text_splitter.split_text(text)
            print(f"PDF split into {len(chunks)} chunks.")

            self.client.add(collection_name=self.collection_name,
                            documents=chunks, ids=list(range(len(chunks))))
            print(f"Indexed {len(chunks)} chunks into Qdrant collection '{self.collection_name}'.")
            print("RAG setup complete.")
        except Exception as e:
            print(f"Error during RAG setup: {e}")
            raise
    
    def _run(self, query: str) -> str:
        """
        Performs a search on the indexed PDF document.
        """
        search_results = self.client.query(
            collection_name=self.collection_name,
            query_text=query,
            limit=5  # Limit to top 5 results
        )

        result_texts = [hit.document for hit in search_results]
        return "\n\n---\n".join(result_texts)

