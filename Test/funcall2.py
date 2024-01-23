import json
from openai import OpenAI

client = OpenAI(api_key="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")

# A dummy function that always returns the same weather information


def get_current_weather(location, unit="fahrenheit"):
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


def run_conversation():
    messages = [
        {"role": "user", "content": "What's the weather like in Boston?"}]
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
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    response = client.chat.completions.create(model="/Users/mac/Downloads/TinyLlama",
                                              messages=messages,
                                              functions=functions,
                                              function_call="auto")
    response_message = response.choices[0].message

    if response_message.get("function_call"):
        available_functions = {"get_current_weather": get_current_weather}
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(
            response_message["function_call"]["arguments"])
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )

        messages.append(response_message)
        messages.append(
            {"role": "function", "name": function_name, "content": function_response}
        )
        second_response = client.chat.completions.create(model="/Users/mac/Downloads/TinyLlama",
                                                         messages=messages)
        return second_response


print(run_conversation())
