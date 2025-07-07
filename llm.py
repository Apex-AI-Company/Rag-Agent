from typing import Any, Dict, List, Union
import google.generativeai as genai
from crewai import BaseLLM

from config import GOOGLE_API_KEY,GEMINI_MODEL

class GeminiLLM(BaseLLM):
    """
    A custom CrewAI LLM wrapper for Google Gemini.
    """
    def __init__(self, model:str = GEMINI_MODEL):
        super().__init__(temperature=0.2, model=model)
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model_instance = genai.GenerativeModel(model)

    def call( self, prompt: Union[str, List[Dict[str,str]]],**kwargs: Any) -> str:
        """
        Handles the API call to Google Gemini.
        """
        if isinstance(prompt,list):
            prompt_text = prompt[-1]['content']
        else:
            prompt_text = prompt

        try:
            response = self.model_instance.generate_content(prompt_text)
            return response.text
        except Exception as e:
            print(f"Error during API call: {e}")
            return "An error occurred while processing your request."
    
    def get_context_window_size(self) -> int:
        """
        Returns the context window size for the model.
        """
        return 32768
    
    def supports_function_calling(self) -> bool:
        """
        Indicates whether the model supports function calling.
        """
        return False
        