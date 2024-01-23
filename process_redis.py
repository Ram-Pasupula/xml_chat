from langchain.llms import HuggingFacePipeline
from constants import (
    MODEL_ID,
    MAX_NEW_TOKENS,
    SERVER_URL,
    DEVICE,
    TORCH_DTYPE,
    base_prompt

)
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredXMLLoader
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from ctransformers import AutoModelForCausalLM
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from utils.embedings import HuggingFaceRedisEmbeddings
from utils.gen_prompts import Prompts
import os
import glob
import re
import streamlit as st
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
    logger.info("loading  embeddinga: ")
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
        # temperature=0.2,
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
            chunks = get_text_chunks_langchain(python_documents)
            Redis.drop_index(
                index_name="ccd_xml", delete_documents=True, redis_url="redis://localhost:6379"
            )
            rds = Redis.from_texts(
                chunks,
                huggingface_ef,
                # metadatas=metadata,
                redis_url="redis://localhost:6379",
                index_name="ccd_xml",
            )
            prompt = st.text_input("Ask Question about your PDF:")
            if prompt:
                if re.search(Prompts.DEMOGRAPHIC.name, prompt, re.IGNORECASE):
                    system_prompt = Prompts.DEMOGRAPHIC.value

                elif re.search(Prompts.PROBLEM.name, prompt, re.IGNORECASE):
                    system_prompt = Prompts.PROBLEM.value

                prompt = base_prompt.format(system_prompt=system_prompt,
                                            user_prompt=prompt)
                print(prompt)
                # print(system_prompt)
                # PROMPT_TEMPLATE = f'''[INST] <<SYS>>
                #                         {system_prompt}
                #                         <</SYS>>
                #                         {prompt}[/INST]
                #                     '''
                # docs = rds.similarity_search(prompt, k)
                # chain = load_qa_chain(llm, chain_type="stuff")
                # response = chain.run(
                #     input_documents=docs, question=PROMPT_TEMPLATE)
                retriever = rds.as_retriever(
                    search_type="similarity", search_kwargs={"k": 2})
                # create a chain to answer questions
                qa = RetrievalQA.from_chain_type(
                    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

                response = qa(prompt)
                print(response)
                st.write(response['result'])
                # st.write(response[0]["result"])
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
