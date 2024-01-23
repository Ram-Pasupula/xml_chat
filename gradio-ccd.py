import os
import glob
import re
import gradio as gr
from langchain.llms import HuggingFacePipeline
from constants import MODEL_ID, MAX_NEW_TOKENS, SERVER_URL, DEVICE, TORCH_DTYPE, base_prompt
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredXMLLoader
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.chains import RetrievalQA
from utils.embedings import HuggingFaceRedisEmbeddings
from utils.gen_prompts import Prompts

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

docs = []
logger.info(f"device :{DEVICE}")
logger.info(f"torch_dtype :{TORCH_DTYPE}")
k = 1
system_prompt = ""
prompt = ""


def load_model():
    model_name = MODEL_ID
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_embeddings_fn():
    logger.info("loading  embeddings: ")
    return HuggingFaceRedisEmbeddings(url=SERVER_URL)


def get_text_chunks_langchain(python_documents):
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=300, chunk_overlap=5
    )
    docs = [x.page_content for x in python_splitter.split_documents(
        python_documents)]
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
    logger.info("llm loaded ")
    return HuggingFacePipeline(pipeline=pipe)


llm = get_llm()
huggingface_ef = get_embeddings_fn()


def ask_question(prompt):
    global system_prompt
    global k
    
    if re.search(Prompts.DEMOGRAPHIC.name, prompt, re.IGNORECASE):
        system_prompt = Prompts.DEMOGRAPHIC.value

    elif re.search(Prompts.PROBLEM.name, prompt, re.IGNORECASE):
        system_prompt = Prompts.PROBLEM.value

    prompt = base_prompt.format(system_prompt=system_prompt, user_prompt=prompt)
    retriever = rds.as_retriever(
        search_type="similarity", search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    response = qa(prompt)
    return response['result']


def process_file(file):
    if file is not None:
        file_location = f"/tmp/ccd/{file.name}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.read())
        file_path = os.path.dirname(file_location)
        logger.info(f"{file_path}/{file.name}")
        logger.info(f"file_location: {file_location}")
        loader = UnstructuredXMLLoader(file_location)
        python_documents = loader.load()
        chunks = get_text_chunks_langchain(python_documents)
        Redis.drop_index(
            index_name="ccd_xml", delete_documents=True, redis_url="redis://localhost:6379"
        )
        global rds
        rds = Redis.from_texts(
            chunks,
            huggingface_ef,
            redis_url="redis://localhost:6379",
            index_name="ccd_xml",
        )
        return "File uploaded successfully."


iface = gr.Interface(fn=ask_question, inputs="text", outputs="text")

file_uploader = gr.Image(
    type="file",
    label="Upload your XML",
    preprocess=process_file
)

iface.add_input(file_uploader)
iface.launch()
