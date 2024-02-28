
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import UnstructuredXMLLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis
from utils.embedings import HuggingFaceRedisEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging

from constants import (
    MODEL_ID,
    MAX_NEW_TOKENS,
    SERVER_URL,
    DEVICE,
    TORCH_DTYPE
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')


def load_model():
    model_name = MODEL_ID
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_embeddings_fn():
    logger.info("Loading embeddings...")
    return HuggingFaceRedisEmbeddings(url=SERVER_URL)


def get_text_chunks_langchain(python_documents):
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=300, chunk_overlap=5
    )
    docs = [x.page_content for x in python_splitter.split_documents(python_documents)]
    return docs


def get_llm():
    model, tokenizer = load_model()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        repetition_penalty=1,
        return_full_text=False,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
        pad_token_id=tokenizer.eos_token_id
    )
    logger.info("Language model loaded.")
    return HuggingFacePipeline(pipeline=pipe)


def file_upload(file_input):
    global rds
    if file_input is not None:
        huggingface_ef = get_embeddings_fn()
        file_location = f"/tmp/{file_input.name}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file_input.read())
        loader = UnstructuredXMLLoader(file_location)
        python_documents = loader.load()
        chunks = get_text_chunks_langchain(python_documents)
        Redis.drop_index(
            index_name="text_file", delete_documents=True, redis_url=SERVER_URL
        )
        rds = Redis.from_texts(
            chunks,
            huggingface_ef,
            redis_url=SERVER_URL,
            index_name="text_file",
        )
        return "Text file uploaded successfully."
    
    
    
def get_tgi_llm():
    from langchain_community.llms import HuggingFaceTextGenInference
    llm = HuggingFaceTextGenInference(
        inference_server_url="http://localhost:8080/",
        max_new_tokens=1024,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.01,
        repetition_penalty=1.03,
    )
    logger.info("Language model loaded.")
    return llm