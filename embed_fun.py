from langchain.llms import HuggingFacePipeline
from constants import (
    MAX_NEW_TOKENS,
    SERVER_URL,
    MODEL_PATH,
)
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredXMLLoader
import os
import faiss
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from embedings import HuggingFaceRedisEmbeddings
model_name = MODEL_ID
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
huggingface_ef = HuggingFaceRedisEmbeddings(url=SERVER_URL)
# model = model.to(device)

docs = []
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=MAX_NEW_TOKENS,
    # temperature=0.2,
    repetition_penalty=1,
    return_full_text=False,
    # device=device
)
llm = HuggingFacePipeline(pipeline=pipe)

docs = []


def get_text_chunks_langchain(python_documents):
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=300, chunk_overlap=0
    )
    docs = [x.page_content for x in python_splitter.split_documents(
        python_documents)]
    # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # docs = [Document(page_content=x)
    return docs


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask Your PDF")

   # xml = st.file_uploader("Upload your pdf", type="xml")

    xml = "/Users/mac/Documents/ws/AI/pdf-qa/SOURCE_DOCUMENTS/CCD.xml"
    # st.write(pdf)
    if xml is not None:
        loader = UnstructuredXMLLoader(xml)
        python_documents = loader.load()
        chunks = get_text_chunks_langchain(python_documents)

        # create embedding

        embeddings = HuggingFaceRedisEmbeddings(url=SERVER_URL)

        knowledge_base = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask Question about your PDF:")
        if user_question:
            query_embedding = huggingface_ef.embed_query(user_question)
            
            docs = knowledge_base.similarity_search(user_question)
            # llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":5,
            #                                           "max_length":64})
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.write(response)


        # st.write(chunks)
if __name__ == '__main__':
    main()
