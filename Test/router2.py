import transformers
import torch
import json
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextStreamer

model_id = "/Users/mac/Downloads/TinyLlama"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=DEVICE, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
def get_employee(id):
    """Retrieve get_employee details"""

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


signature_emp = {
        "name": "get_employee",
        "description": "Get the employee details by id",
        "arguments": [
            {
                "name": "id",
                "type": "string",
                "description": "The employee id string"
            }
        ]
         
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

functionList = ''
functionList += json.dumps(signature_emp, indent=4, separators=(',', ': '))
functionList += json.dumps(signature_emp, indent=4, separators=(',', ': '))
functionList += json.dumps(search_bing_metadata, indent=4, separators=(',', ': '))

print(functionList)

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
    response = model.generate(**inputs, streamer=streamer,  max_new_tokens=200)
    if response["name"] == "get_employee":
            headlines = get_employee(1)
            print(headlines)  
   
    



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
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=500)  
    
    
#stream('Search bing for the tallest mountain in Ireland')   
stream('get Jacksonville weather today')  



# import time

# # Initialize the prompt
# prompt = "Once upon a time"

# # Record the start time
# start_time = time.time()

# # Encode the prompt
# input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

# # Set the maximum length for generation
# max_length = 100

# # Generate the sequence
# output = model.generate(input_ids, max_length=max_length)

# # Record the end time
# end_time = time.time()

# # Calculate the tokens per second
# elapsed_time = end_time - start_time
# tokens_per_sec = max_length / elapsed_time

# print(f"Tokens per second: {tokens_per_sec}")   