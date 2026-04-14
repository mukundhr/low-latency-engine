import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("LLM_MODEL", "mistral")


class LLM:
    def __init__(self):
        self.url = OLLAMA_URL
        self.model = MODEL_NAME

        print(f"Using LLM: {self.model}")
        print(f"Ollama URL: {self.url}")

    # ✅ NORMAL GENERATION (for /query)
    def generate(self, prompt, num_predict=120):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": num_predict
            }
        }

        response = requests.post(self.url, json=payload)
        result = response.json()

        return result.get("response", "")

    # ✅ STREAMING (for /stream)
    def stream_generate(self, prompt, num_predict=120):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": num_predict
            }
        }

        response = requests.post(self.url, json=payload, stream=True)

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get("response", "")