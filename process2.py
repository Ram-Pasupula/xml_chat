from langchain.llms import HuggingFacePipeline
from constants import (
    MODEL_ID,
    MAX_NEW_TOKENS,
    SERVER_URL,
    DEVICE,
    TORCH_DTYPE
)
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredXMLLoader
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from utils.embedings import HuggingFaceRedisEmbeddings
import os
import glob
import streamlit as st

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

docs = []


class ProcessClass(object):
    @staticmethod
    def load_model():
        model_name = MODEL_ID
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    @staticmethod
    def get_embeddings_fn():
        logger.info("loading  embeddinga: ")
        return HuggingFaceRedisEmbeddings(url=SERVER_URL)

    @staticmethod
    def get_text_chunks_langchain(python_documents):
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=300, chunk_overlap=5
        )
        docs = [x.page_content for x in python_splitter.split_documents(
            python_documents)]
        return docs

    @staticmethod
    def get_llm():
        model, tokenizer = ProcessClass.load_model()
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=MAX_NEW_TOKENS,
            repetition_penalty=1,
            return_full_text=False,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
        logger.info("llm loaded ")
        return HuggingFacePipeline(pipeline=pipe)


llm = ProcessClass.get_llm()
huggingface_ef = ProcessClass.get_embeddings_fn()


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your XML")
    st.header("Ask Your XML")

    xml = st.file_uploader("Upload your XML", type="xml")

    try:
        if xml is not None:
            file_location = f"/tmp/ccd/{xml.name}"
            with open(file_location, "wb+") as file_object:
                file_object.write(xml.read())
            file_path = os.path.dirname(file_location)
            logger.info(f"{file_path}/{xml.name}")
            logger.info(f"file_location: {file_location}")
            loader = UnstructuredXMLLoader(file_location)
            python_documents = loader.load()
            chunks = ProcessClass.get_text_chunks_langchain(python_documents)
            knowledge_base = FAISS.from_texts(chunks, huggingface_ef)
            user_question = st.text_input("Ask Question about your PDF:")
            # Get user input iteratively
            if user_question:
                try:
                    docs = knowledge_base.similarity_search(user_question)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(
                        input_documents=docs, question=user_question)
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

            # Add a Clear Response or Reset Button
            if st.button("Clear Response"):
                st.empty()  # Clear the response

            # Optionally, you can also add a button to reset the input
            if st.button("Reset Input"):
                user_question = st.text_input(
                    "Ask a question", value="")  # Reset the input
    except Exception:
        raise Exception(status_code=500, detail='File not able to load')
    finally:
        try:
            files = glob.glob(f"/tmp/ccd/{xml.name}")
            for f in files:
                os.remove(f)
        except Exception:
            pass
        else:
            logger.info("Successfully deleted temp files")


if __name__ == '__main__':
    main()
