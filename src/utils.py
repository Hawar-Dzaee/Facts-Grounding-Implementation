
import json 


def fill_out_prompt(prompt_template, **kwargs):
    """Formats a prompt template with {{variable}} style placeholders"""
    for key, value in kwargs.items():
        placeholder = f"{{{{{key}}}}}"
        prompt_template = prompt_template.replace(placeholder, str(value))
    return prompt_template



def dump_in_jsonl(data,file_name):
    with open(file_name,'a') as file : 
        json.dump(data,file)
        file.write("\n") 