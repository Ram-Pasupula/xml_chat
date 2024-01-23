St= """<s> ### Call:\n[\n    {\n        "name": "sunset",\n        "description": "Get the time of sunset in a city",\n        "parameters": [\n            {\n                "name": "city"\n            },\n            {\n                "name": "date",\n                "type": "str"\n            }\n        ],\n        "required": [\n            "city"\n        ]\n    },\n    {\n        "name": "get_current_weather",\n        "description": "Get the current weather in a given location",\n        "parameters": [\n            {\n                "name": "location"\n            },\n            {\n                "name": "unit",\n                "type": "str"\n            }\n        ],\n        "required": [\n            "location"\n        ]\n    }\n]\n### User: Get current weather Jacksonville, Florida\n```\n\nIn this example, the user inputs the city name "Jacksonville, Florida" and the unit "Fahrenheit". The script will return the current weather conditions for the given location.</s>'"""

text = """### Call:\n[\n    {\n        "name": "sunset",\n        "description": "Get the time of sunset in a city",\n        "parameters": [\n            {\n                "name": "city"\n            },\n            {\n                "name": "date",\n                "type": "str"\n            }\n        ],\n        "required": [\n            "city"\n        ]\n    },\n    {\n        "name": "get_current_weather",\n        "description": "Get the current weather in a given location",\n        "parameters": [\n            {\n                "name": "location"\n            },\n            {\n                "name": "unit",\n                "type": "str"\n            }\n        ],\n        "required": [\n            "location"\n        ]\n    }\n]\n### User: Get current weather Jacksonville, Florida\n'"""

a = St.replace(text, "").replace("<s>", "")
print(a)
b = a.replace("<\\s>", "").strip("\n").strip()
print(b)