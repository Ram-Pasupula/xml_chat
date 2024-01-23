from local_llm_function_calling import Generator
import json
import os
import requests
from local_llm_function_calling import Constrainer, JsonSchemaConstraint
from local_llm_function_calling.model.huggingface import HuggingfaceModel
from transformers import AutoTokenizer, StoppingCriteria, GenerationConfig, AutoModelForCausalLM, TextStreamer
from peft import AutoPeftModelForCausalLM
import colorama
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

def get_top_headlines(id):
    """Retrieve top headlines from newsapi.org (API key required)"""

    base_url = f"https://dummy.restapiexample.com/api/v1/employee/{id}"
    print(base_url)

    response = requests.get(base_url)
    data = response.json()

    if data['status'] == 'success':
        print(f"Processing {data['data']} articles from employees")
        return json.dumps(data['data'])
    else:
        print("Request failed with message:", data['message'])
        return 'No articles found'


signature_get_employee_details = {
    "name": "get_employee_details",
    "description": "get employee salary",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "get employee salary for.",
            }
        },
        "required": ["id"],
    }
}
search_bing_metadata = {
    "function": "search_bing",
    "description": "Search the web for content on Bing. This allows users to search online/the internet/the web for content.",
    "arguments": [
        {
            "name": "query",
            "type": "string",
            "description": "The search query string"
        }
    ]
}
# Define a function and models
functions = [
    {
        "name": "get_employee",
        "description": "Get the employee details by id",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "the employee id",
                    "maxLength": 20,
                },
                "unit": {"type": "string", "enum": ["id"]},
            },
            "required": ["id"],
        },
    },
     {
        "name": "calculate_age",
        "description": "Calculates the age of a person.",
        "parameters": [
            {
                "name": "birth_date",
                "type": "date",
                "description": "The date of birth of the person."
            },
            {
                "name": "current_date",
                "type": "date",
                "description": "The current date."
            }
        ],
        "required": ["birth_date", "current_date"]
    }
]
model_id = "/Users/mac/Downloads/TinyLlama"
# Initialize the generator with the Hugging Face model and our functions
model = AutoModelForCausalLM.from_pretrained(model_id) # this can easily exhaust Colab RAM. Note that bfloat16 can't be used on cpu.

#generator = Generator.hf(functions, model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
functionList = ''

functionList += json.dumps(search_bing_metadata, indent=4, separators=(',', ': '))
functionList += json.dumps(signature_get_employee_details, indent=4, separators=(',', ': '))


# Define a stream *with* function calling capabilities
def stream(user_prompt):
  # Define the roles and markers
    B_INST, E_INST = "[INST]", "[/INST]"
    B_FUNC, E_FUNC = "<FUNCTIONS>", "</FUNCTIONS>\n\n"

    # Format your prompt template
    prompt = f"{B_FUNC}{functionList.strip()}{E_FUNC}{B_INST} {user_prompt.strip()} {E_INST}\n\n"

    inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)

    streamer = TextStreamer(tokenizer)

    # Despite returning the usual output, the streamer will also print the generated text to stdout.
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=200)

# Define a stream *without* function calling capabilities
def stream2(user_prompt):
    system_prompt = 'You are a helpful assistant that provides accurate and concise responses'

    # Guanaco style
    B_INST, E_INST = "### Human:", "### Assistant:"
    prompt = f"{B_INST} {user_prompt.strip()}{E_INST}\n\n"

    # # Llama style
    # B_INST, E_INST = "[INST]", "[/INST]"
    # B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    # prompt = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n"

    inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)

    streamer = TextStreamer(tokenizer)

    # Despite returning the usual output, the streamer will also print the generated text to stdout.
    model.generate(**inputs, streamer=streamer, max_new_tokens=500)

    
stream(' get employee details')    

# Generate text using a prompt
#response = generator.generate("How old am I, birthdate 06-11-1983")
# print(response)
# # call functions requested by the model
# if response["name"] == "get_employee":
#     headlines = get_top_headlines(1)
#     print(headlines)    
# else :
#     print(response)    
     
     
        
