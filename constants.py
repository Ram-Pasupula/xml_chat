import os
from chromadb.config import Settings
from langchain.document_loaders import CSVLoader, UnstructuredXMLLoader
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"
EMBED_CACHE_FOLDER = f"{ROOT_DIRECTORY}/embed"
MODELS_PATH = "./models"
SERVER_URL = "http://localhost:9081/embed"
STORE_NAME = "emb_store"
MODEL_PATH = "/Users/mac/Documents/ws/AI/rag-api/models"
MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_ID = "wy90021/llama-2-7b-chat-health"
MODEL_ID = "dim/tiny-llama-2T-open-orca-ru-10000-step"
# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 6

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# Context Window and Max New Tokens
CONTEXT_WINDOW_SIZE = 2048
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

# If you get a "not enough space in the buffer" error, you should reduce the values below, start with half of the original values and keep halving the value until the error stops appearing

N_GPU_LAYERS = 0  # Llama-2-70B has 83 layers
N_BATCH = 512


DOCUMENT_MAP = {
     ".xml":  UnstructuredXMLLoader,

    ".csv": CSVLoader
}
