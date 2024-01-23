import gradio as gr
from langchain.document_loaders import UnstructuredXMLLoader
from langchain.vectorstores.redis import Redis
from langchain.chains import RetrievalQA
from utils.gen_prompts import Prompts
import logging
import re
import time
import os
from constants import (
    REDIS_URL,
    base_prompt
)
from utils.utility import load_model, get_embeddings_fn, get_text_chunks_langchain, get_llm, get_embeddings_fn
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
k = 1
system_prompt = ""
prompt = ""
rds = None

llm = get_llm()
huggingface_ef = get_embeddings_fn()


def loading_pdf():
    return "Loading..."


def upload_file(filepath):
    return filepath


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def create_index(file_input):
    global rds
    if file_input is not None:
        audio_file_name = file_input.split("/")[-1]
        files = {'file': (audio_file_name, open(file_input, 'rb'))}
        file_location = f"/tmp/{audio_file_name}"
        with open(file_location, "wb+") as file_object:
            file_object.write(files['file'][1].read())
        loader = UnstructuredXMLLoader(file_location)
        python_documents = loader.load()
        chunks = get_text_chunks_langchain(python_documents)
        Redis.drop_index(
            index_name="ccd_xml", delete_documents=True, redis_url=REDIS_URL
        )
        rds = Redis.from_texts(
            chunks,
            huggingface_ef,
            redis_url=REDIS_URL,
            index_name="ccd_xml",
        )
        return "CCD file uploaded successfully."


def chat(chat_history, question):
    global system_prompt
    global k
    global rds

    if re.search(Prompts.DEMOGRAPHIC.name, question, re.IGNORECASE):
        system_prompt = Prompts.DEMOGRAPHIC.value

    elif re.search(Prompts.PROBLEM.name, question, re.IGNORECASE):
        system_prompt = Prompts.PROBLEM.value

    prompt = base_prompt.format(
        system_prompt=system_prompt, user_prompt=question)
    retriever = rds.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False
    )
    bot_response = qa(prompt)
    response = ""
    # [bot_response[i:i+1] for i in range(0, len(bot_response), 1)]:
    for letter in ''.join(bot_response['result'].split("Question")[0]):
        response += letter + ""
        yield chat_history + [(question, response)]


with gr.Blocks(css="footer{display:none !important}", title="ASR",
               head="CCD- chat box",) as demo:
    gr.Markdown('Chat with CCD file')    
    with gr.Column(elem_id="col-container"):
       # gr.HTML(title)
        with gr.Column():
            pdf_doc = gr.File(label="Load a CCD file", file_types=['.xml'], type="filepath")
            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Load xml to process")
        
        chatbot = gr.Chatbot([], elem_id="chatbot")
        question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send Message")
    load_pdf.click(loading_pdf, None, langchain_status, queue=False)    
    load_pdf.click(create_index, inputs=[pdf_doc], outputs=[langchain_status], queue=False)
    question.submit(chat, [chatbot, question], chatbot)
    submit_btn.click(chat, [chatbot, question], chatbot)

#demo.launch()    

demo.queue().launch(debug=True)
