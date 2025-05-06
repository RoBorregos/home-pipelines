# Local LLM
API_KEY = "ollama"
# BASE_URL = "http://10.22.131.22:11434/v1"
BASE_URL = "http://192.168.31.10:11434/v1"

MODEL = "qwen3:4b"

TEMPERATURE = 1.5
# TEMPERATURE = 0.0

# OpenAI API, used for batch generation
import os
from dotenv import load_dotenv 

# load_dotenv()

# API_KEY = os.getenv('API_KEY')
# # BASE_URL = None
# MODEL = "gpt-4.1-nano"
