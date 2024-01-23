import os

from langchain.document_loaders import UnstructuredXMLLoader
import torch
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

logger.info(f"device :{DEVICE}")
logger.info(f"torch_dtype :{TORCH_DTYPE}")

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/"
MODELS_PATH = "./models"
SERVER_URL = "http://localhost:9081/embed"
REDIS_URL = "redis://localhost:6379"
STORE_NAME = "emb_store"
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_ID = "wy90021/llama-2-7b-chat-health"

# MODEL_ID = "dim/tiny-llama-2T-open-orca-ru-10000-step"

MODEL_ID = "/Users/mac/Downloads/TinyLlama"
# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 6
base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
# Define the Chroma settings

# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = 2048
MAX_NEW_TOKENS = 700 # int(CONTEXT_WINDOW_SIZE/4)

N_GPU_LAYERS = 0  # Llama-2-70B has 83 layers
N_BATCH = 512


DOCUMENT_MAP = {
    ".xml":  UnstructuredXMLLoader
}

