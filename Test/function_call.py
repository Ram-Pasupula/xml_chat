import json
import inspect
from transformers import AutoTokenizer, StoppingCriteria, GenerationConfig, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import colorama
model_id = "/Users/mac/Downloads/TinyLlama"
base_model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


class Streamer:
    def __init__(self, enabled):
        self.do_stream = enabled

    def put(self, input_ids):
        if self.do_stream:
            print(tokenizer.decode(input_ids[0]))

    def end(self):
        return


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generation = self.tokenizer.decode(input_ids[0])
        new_generation = decoded_generation[self.start_length:]
        return any(
            [
                eof_string in new_generation[len(eof_string):]
                for eof_string in self.eof_strings
            ]
        )


config = GenerationConfig.from_model_config(base_model.config)
config.max_length = 4096
config.max_new_tokens = 128


def infer(text):
    """
    Infer a function call from a prompt
    :param text: The prompt to infer from. Should not end with a header (eg, "### System: Hi!\n" but not "### System:")
    :return: The new text, and the header it starts with
    """

    headers = ["### System:", "### User:", "### Call:", "### Return:", "### Assistant:"]
    ids = base_model.generate(
        input_ids=tokenizer.encode(text, return_tensors="pt").to(base_model.device),
        stopping_criteria=[EndOfFunctionCriteria(len(text), headers)],
        generation_config=config,
    )
    decoded_generation = tokenizer.decode(ids[0])
    text = f"""{text}"""
    x = decoded_generation.replace(text, "").replace("<s>", "")
    new_generation = x.replace("<\\s>", "").replace("\n", " ").strip()

    # Find the header the response starts with
    start_header = ""
    for header in headers:
        if new_generation.startswith(header):
            start_header = header
            break

    if start_header != "":
        new_generation = new_generation[len(start_header):]

    # Find the header the response ends with
    for header in headers:
        if new_generation.endswith(header):
            new_generation = new_generation[:-len(header)]

    return new_generation, start_header


functions = []
previous_functions = []


def register_function(json_definition, fn):
    functions.append((json_definition, fn))


running_chat = ""


def query(prompt):
    """
    Main function for querying the assistant
    :param prompt: The prompt to query with
    """
    global running_chat
    global functions
    global previous_functions

    if prompt == "reset":
        running_chat = ""
        previous_functions = []
        return "Resetting chat"

    prompt = "### User: " + prompt + "\n"

    # If functions changed, create a new system text
    if functions != previous_functions:
        system_text = "### Call:\n" + json.dumps([f[0] for f in functions], indent=4) + "\n"
        running_chat += system_text
        previous_functions = functions

    # Generate the next response
    output, role = infer(running_chat + prompt)

    # If the response is a function call, try to call it
    if role == "### Call:":
        try:
            call = json.loads(output)
            if "name" not in call or "parameters" not in call:
                raise ValueError
        except ValueError:
            print(colorama.Fore.RED + "Error parsing call: " + output + colorama.Style.RESET_ALL)
            response = "Assistant: The call I wrote was formatted wrong. Please try again.\n"
            running_chat += response
            return response

        # Check if the function exists
        if call["name"] not in [f[0]["name"] for f in functions]:
            print(colorama.Fore.RED + "Error: function " + call["name"] + " does not exist" + colorama.Style.RESET_ALL)
            response = "### Assistant: The function I wrote does not exist. Please try again.\n"
            running_chat += response
            return response

        # Try to call the function
        try:
            fn = [f[1] for f in functions if f[0]["name"] == call["name"]][0]
            params = call["parameters"].values()

            fn_output = str(fn(*params))
            params_str = ", ".join([str(p) for p in params])
            print(f"{colorama.Fore.GREEN}{call['name']}({params_str}) = {fn_output}{colorama.Style.RESET_ALL}")

            fn_output = {
                "result": fn_output,
            }
            fn_output = json.dumps(fn_output)

        except Exception as e:
            print(colorama.Fore.RED + "Error calling function: " + str(e) + "\n" + str(call) + colorama.Style.RESET_ALL)
            response = """### Assistant: The function tried to call had an error. Please try again.\n"""
            running_chat += response
            return response

        # Feed the return value back into the model
        running_chat += f"""### Call:\n {json.dumps(call)} \n"""
        running_chat += f"""### Return:\n {fn_output} \n"""
        output, role = infer(running_chat)

    # Add the response to the running chat
    running_chat += f"""### Assistant: {output} \n"""

    return output.strip()


def ai_callable(func):
    def wrap(*args, **kwargs):
        return func(*args, **kwargs)

    definition = {
        "name": func.__name__,
        "description": func.__doc__.strip(),
        "parameters": [],
        "required": []
    }
    parameters = definition["parameters"]
    required = definition["required"]
    signature = inspect.signature(func)

    for k, v in signature.parameters.items():
        parameter = {
                "name": k
        }
        if v.annotation is not inspect.Parameter.empty:
            parameter["type"] = v.annotation.__name__
        if v.default is inspect.Parameter.empty:
            required.append(k)
        parameters.append(parameter)

    register_function(definition, func)
    return wrap

        
if __name__ == "__main__":
    @ai_callable
    def sunset(city, date: str = "2021-01-01"):
        """
        Get the time of sunset in a city
        """
        return "8:00 PM"
    
    # @ai_callable
    # def get_current_weather(location, unit: str = "celsius"):
    #     """
    #     Get the current weather in a given location
    #     """
    #     return """{ [15, "celsius", "Clear", 3.6, "NW", 0.67]}"""

    while True:
        print(query(input("> ")))
