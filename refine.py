import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Gemini API (hidden configuration)
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def process_text_enhancement(original_text):
    """
    Internal function to refine and enhance text using Gemini
    
    Args:
        original_text (str): Original text to be refined
    
    Returns:
        str: Refined text or original text if refinement fails
    """
    prompt = f"""Refine this text to ensure:
    - Perfect grammatical coherence
    - No unnecessary repetitions
    - Crisp, clear language
    - Maintain original core meaning
    - Professional and concise tone

    Original Text:
    {original_text}

    Refined Text:
    """

    try:
        # Use Gemini Pro model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate refined text
        response = model.generate_content(prompt)
        refined_text = response.text.strip()
        
        return refined_text
    except Exception as e:
        logging.error(f"Text Enhancement Error: {e}")
        return original_text