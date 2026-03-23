import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
HTTP_PROXY = os.getenv("HTTP_PROXY", "http://127.0.0.1:7897")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "http://127.0.0.1:7897")

WORKER_NAME = "qwen-sam-worker"
WORKER_PORT = 8002
SAM2_MODEL_PATH = os.getenv("SAM2_MODEL_PATH", "/home/tb5z035i/workspace/checkpoints/facebook/sam2-hiera-large")
