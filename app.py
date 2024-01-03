import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import faiss
import sys
import torch
import numpy as np
import logging
from embedings import HuggingFaceRedisEmbeddings
import pandas as pd
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders import DataFrameLoader
from load_fe_models import load_mps_model, load_o_model, load_fe_model
from style_util import (change_label_style, TOOL_HIDE, get_footer,
                        get_page_conf, label, title, side_foot)
from transformers import (GenerationConfig, pipeline)
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredXMLLoader
from transformers.data.processors.squad import SquadExample
from constants import (
    MAX_NEW_TOKENS,
    SERVER_URL,
    MODEL_PATH,
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot.
    ''')
    st.write('Contact Analytics')

load_dotenv()
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model_name = "wy90021/llama-2-7b-chat-health"
model_name = "dim/tiny-llama-2T-open-orca-ru-10000-step"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
huggingface_redis_embedding = HuggingFaceRedisEmbeddings(
    url=SERVER_URL)
#model = model.to(device)

docs = []
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=MAX_NEW_TOKENS,
    #temperature=0.2,
    repetition_penalty=1,
    return_full_text=False,
    #device=device
)


def get_text_chunks_langchain(python_documents):
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=300, chunk_overlap=0
    )
    docs = [x.page_content for x in python_splitter.split_documents(
        python_documents)]
    # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def main():
    st.header("Chat with XML 💬")

    # upload a PDF file
    # xml = st.file_uploader("Upload your XML", type='pdf')
    # xml = st.file_uploader("Upload your XML")
    xml = "/Users/mac/Documents/ws/AI/pdf-qa/SOURCE_DOCUMENTS/CCD.xml"
    # st.write(pdf)
    if xml is not None:
        logging.info("Loading XML documents")
        loader = UnstructuredXMLLoader(xml)
        python_documents = loader.load()
        chunks = get_text_chunks_langchain(python_documents)

        text_embeddings = np.array(
            huggingface_redis_embedding.embed_documents(chunks))
        index = faiss.IndexFlatL2(text_embeddings.shape[1])  # L2 distance
        index.add(text_embeddings)
        # Accept user questions/query
        query = st.text_input("Ask questions about your XML file:")

        if query:
            print(f"query:{query}")
            # Example query
            query_embedding = np.array(
                huggingface_redis_embedding.embed_query(query))
            k = 1  # Number of nearest neighbors to retrieve
            distances, indices = index.search(
                query_embedding.reshape(1, -1), k)
            print("Nearest neighbors indices:", indices)
            print("Distances to nearest neighbors:", distances)
            # Retrieve the top-k documents
            top_k_documents = [python_documents[idx]
                               for idx in indices.flatten()]
            # print(f"top_k_documents: {top_k_documents}")
            context = " "
            for document in top_k_documents:
                context += document.page_content

            # squad_example = SquadExample(
            #     qas_id="1",
            #     question_text=query,
            #     context_text=context,
            #     answer_text="",
            #     start_position_character=0,  # Adjust as needed
            #     title="CCD"  # Adjust as needed
            # )
            
            input_text = f"{context} {query}"
            answer = pipe(input_text)
            # print(f"Question: {query}")
            print(f"Answer: {answer[0]['generated_text']}")
            st.write(answer[0])
            # Generate text based on the combined input
# generated_text = text_generator(input_text, max_length=100, num_return_sequences=1)

# print("Generated Text:")
# print(generated_text[0]['generated_text'])


if __name__ == '__main__':
    main()
