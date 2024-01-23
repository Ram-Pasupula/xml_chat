from local_llm_function_calling import Generator
import json
import os
import requests
from local_llm_function_calling import Constrainer, JsonSchemaConstraint
from local_llm_function_calling.model.huggingface import HuggingfaceModel

# Define a function and models
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                    "maxLength": 20,
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_employeer",
        "description": "Get the employee details",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "employee id eg:1",
                    "maxLength": 20,
                }
            },
            "required": ["id"],
        },
    }
]
model_id = "/Users/mac/Downloads/TinyLlama"

generator = Generator(functions, HuggingfaceModel(model_id))
function_call = generator.generate("What is current  weather in Jacksonville?")

print(function_call)