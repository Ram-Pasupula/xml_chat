from semantic_router import RouteLayer
from semantic_router.encoders import HuggingFaceEncoder,OpenAIEncoder
from datetime import datetime
from zoneinfo import ZoneInfo
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from ctransformers import AutoModelForCausalLM
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from utils.embedings import HuggingFaceRedisEmbeddings
from langchain.llms import HuggingFacePipeline
from semantic_router import Route
from semantic_router.utils.function_call import get_schema
import os
import openai
os.environ["OPENAI_API_KEY"] = "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"
openai_api_key = "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"
MODEL_ID = "/Users/mac/Downloads/TinyLlama"


def load_model():
    model_name = MODEL_ID
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def get_llm():
    model, tokenizer = load_model()
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        # temperature=0.2,
        repetition_penalty=1,
        return_full_text=False,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=pipe)
def get_embeddings_fn():
    return HuggingFaceRedisEmbeddings(url="http://localhost:9081/embed")

def get_time(timezone: str) -> str:
    """Finds the current time in a specific timezone.

    :param timezone: The timezone to find the current time in, should
        be a valid timezone from the IANA Time Zone Database like
        "America/New_York" or "Europe/London". Do NOT put the place
        name itself like "rome", or "new york", you must provide
        the IANA format.
    :type timezone: str
    :return: The current time in the specified timezone."""
    now = datetime.now(ZoneInfo(timezone))
    return now.strftime("%H:%M")


time_schema = get_schema(get_time)
print(time_schema)
time = Route(
    name="get_time",
    utterances=[
        "what is the time in new york city?",
        "what is the time in london?",
        "I live in Rome, what time is it?",
    ],
    function_schema=time_schema,
)

politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president" "don't you just hate the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

routes = [politics, chitchat, time]

# print(routes)

encoder = get_embeddings_fn()
llm = get_llm()
#huggingface_ef = get_embeddings_fn()

enable_gpu = True  # offload LLM layers to the GPU (must fit in memory)
model, tokenizer = load_model()
#rl = RouteLayer(encoder=encoder, routes=routes, llm=llm)
from semantic_router.encoders import OpenAIEncoder

rl = RouteLayer(encoder=OpenAIEncoder(openai_api_key=openai_api_key), routes=routes)
# llm = LlamaCppLLM(name="TinyLlama", llm=llm, max_tokens=None)
# rl = RouteLayer(encoder=encoder, routes=routes, llm=llm)
out = rl("what's the time in New York right now?")
print(out)
get_time(**out.function_call)
