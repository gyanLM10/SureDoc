import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ Gemini import

# Load .env variables
load_dotenv()

# Ensure GOOGLE_API_KEY is available
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables or .env file.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


class LLMModel:
    def __init__(self, model_name="gemini-1.5-flash"):
        if not model_name:
            raise ValueError("Model is not defined.")
        self.model_name = model_name
        self.gemini_model = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0,             # deterministic responses
            max_output_tokens=1024     # limit token length
        )

    def get_model(self):
        return self.gemini_model


# Quick test if run directly
if __name__ == "__main__":
    llm_instance = LLMModel()
    llm_model = llm_instance.get_model()
    response = llm_model.invoke("Hello, Gemini!")
    print("Response:", response.content if hasattr(response, "content") else response)

