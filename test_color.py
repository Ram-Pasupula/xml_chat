import json
from colorama import Fore, Style

def convert_to_colored_output(json_list):
    colors = [Fore.BLUE, Fore.GREEN, Fore.RED, Fore.YELLOW, Fore.MAGENTA, Fore.CYAN]
    speaker_colors = {}
    output = ""

    for item in json_list:
        for speaker, text in item.items():
            if speaker not in speaker_colors:
                speaker_colors[speaker] = colors[len(speaker_colors) % len(colors)]

            output += f"{speaker_colors[speaker]}{speaker}:{Style.RESET_ALL} {text}\n"

    return output

json_list = [
    {'SPEAKER_01': 'Hello how are you'},
    {'SPEAKER_02': 'I am good!'},
    {'SPEAKER_01': 'WHERE are you?'}
]

output = convert_to_colored_output(json_list)
print(output)
