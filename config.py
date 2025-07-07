import os 
import sys
from dotenv import load_dotenv
import warnings

# API Config
load_dotenv()
warnings.filterwarnings("ignore")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if GOOGLE_API_KEY is set
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY is not set in the environment variables.")
    sys.exit(1)

# Document and Model Config
PDF_FILE = "apexai.pdf"
GEMINI_MODEL = "gemini-1.5-flash"
QDRANT_COLLECTION_NAME = "rag_collection"

