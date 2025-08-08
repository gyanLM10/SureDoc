import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ Gemini import

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # Optional, but safe

class LLMModel:
    def __init__(self, model_name="gemini-pro"):
        if not model_name:
            raise ValueError("Model is not defined.")
        self.model_name = model_name
        self.gemini_model = ChatGoogleGenerativeAI(model=self.model_name)

    def get_model(self):
        return self.gemini_model

if __name__ == "__main__":
    llm_instance = LLMModel()
    llm_model = llm_instance.get_model()
    response = llm_model.invoke("hi")

    print(response)
