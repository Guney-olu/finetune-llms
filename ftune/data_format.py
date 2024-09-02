import json
import pandas as pd
from datasets import Dataset, DatasetDict

json_file_path = './keyword_fine.json' 

with open(json_file_path, 'r') as file:
    json_data = json.load(file)

def json_to_dataset(json_data):
    flat_data = []
    for entry in json_data:
        input_text = entry["input"]
        output_text = json.dumps(entry["output"]) 
        instruction = "Classify the keywords into relevant and irrelevant categories based on the company profile."
        flat_data.append({"instruction": instruction, "input": input_text, "output": output_text})
        
    return Dataset.from_pandas(pd.DataFrame(flat_data))

dataset = json_to_dataset(json_data)

prompt_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = "[EOS]" 

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = prompt_format.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


def add_text_field(examples):
    texts = formatting_prompts_func(examples)["text"]
    return {"text": texts}

formatted_dataset = dataset.map(add_text_field, batched=True, remove_columns=None)

formatted_dataset_dict = DatasetDict({
    "train": formatted_dataset
})

formatted_dataset_dict.save_to_disk("formatted_dataset")

dataset = formatted_dataset_dict

#print(dataset)
