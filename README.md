# Local Document Q&A with CrewAI and Google Gemini 
## RAG Agent

This project demonstrates a powerful, local question-answering system using a Retrieval-Augmented Generation (RAG) architecture. It leverages CrewAI to orchestrate an AI agent powered by Google Gemini, which answers questions based on the content of a local PDF document.

The core principle is to ground the AI's responses in a verified knowledge source, preventing hallucinations and ensuring the answers are accurate and context-aware.

## Architecture Overview

The system follows a two-stage process:

1.  **Context Retrieval (RAG)**: When a user asks a question, the system first searches a pre-indexed vector database of the PDF document to find the most relevant text chunks.
2.  **Content Synthesis (CrewAI)**: The retrieved context and the original question are then passed to a specialized CrewAI agent. This agent's sole purpose is to synthesize a clear and accurate answer based *only* on the provided information.

```
┌──────────────────┐      ┌─────────────────────────────────┐      ┌───────────────────┐
│  User's Question │ ───> │     SimpleRAGTool (Qdrant)      │ ───> │  Relevant Context │
└──────────────────┘      │ (Finds relevant text in PDF)    │      │   from Document   │
                          └─────────────────────────────────┘      └─────────┬─────────┘
                                                                           │
                                                                           ▼
                                                         ┌───────────────────────────────────┐
                                                         │   CrewAI Agent (Google Gemini)    │
                                                         │ (Synthesizes answer from context) │
                                                         └───────────────────────────────────┘
                                                                           │
                                                                           ▼
                                                                  ┌──────────────────┐
                                                                  │ Final, Grounded  │
                                                                  │      Answer      │
                                                                  └──────────────────┘
```

## Tech Stack

*   **AI Orchestration**: [CrewAI](https://www.crewai.com/)
*   **LLM**: Google Gemini (`gemini-1.5-flash`) via the `google-generativeai` SDK
*   **Vector Database**: [Qdrant](https://qdrant.tech/) (in-memory)
*   **Document Loading**: [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)
*   **Text Processing**: [LangChain](https://www.langchain.com/) (for `RecursiveCharacterTextSplitter`)
*   **Configuration**: `python-dotenv`

## Setup and Installation

**1. Clone the Repository**
```bash
https://github.com/Apex-AI-Company/Rag-Agent.git
cd Rag-Agent
```

**2. Create a Virtual Environment**
It's highly recommended to use a virtual environment to manage dependencies.
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Your API Key**
Create a `.env` file in the root of the project directory and add your Google API key. You can get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

```env
# .env file
GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY_HERE"
```

## How to Run

**1. Add Your Document**
Place the PDF file you want to query into the `knowledge/` directory. The script is currently configured to use `knowledge/Copy of APEX AI company overview.pdf`.

**2. Execute the Script**
Run the main application from your terminal:
```bash
python main.py
```

**3. Ask a Question**
The script will prompt you to enter your question.
```
Please enter your question about the document: What services does APEX AI offer?
```

The crew will then kick off, retrieve the relevant context, and generate a final answer based on the document's contents.

## Configuration

You can easily modify the core settings at the top of the `main.py` script:

*   `PDF_FILE_PATH`: Change this to the path of your PDF document.
*   `GEMINI_MODEL`: Change this to another compatible Gemini model, such as `"gemini-1.5-pro"`.
*   `QDRANT_COLLECTION_NAME`: Modify the name of the in-memory collection if needed.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


