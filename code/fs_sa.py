from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch
import random
import argparse
import csv
import gc
import os
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

SEED = 42

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('-m', type=int, help='Model to evaluate', required=True)

args = parser.parse_args()

if args.m == 0:
    model_path = "google/gemma-2-2b-it"
    model_name = "gemma_2_2b"
elif args.m == 1:
    model_path = "google/gemma-2-9b-it"
    model_name = "gemma_2_9b"
elif args.m == 2:
    model_path = "google/gemma-3-1b-it"
    model_name = "gemma_3_1b"
elif args.m == 3:
    model_path = "meta-llama/Llama-2-7b-chat-hf"
    model_name = "llama_2_7b"
elif args.m == 4:
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = "llama_3_1_8b"
elif args.m == 5:
    model_path = "meta-llama/Llama-3.2-3B-Instruct"
    model_name = "llama_3_2_3b"
elif args.m == 6:
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_name = "qwen_2_5_7b"
elif args.m == 7:
    model_path = "Qwen/Qwen3-8B"
    model_name = "qwen_3_8b"
elif args.m == 8:
    model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_name = "mixtral"
elif args.m == 9:
    model_path = "mistralai/Mistral-7B-Instruct-v0.3"
    model_name = "mistral"

data = pd.read_csv("/data/tomas/hate/dataset.csv")

data_train = data[data['__split'] == 'train']
data_test = data[data['__split'] == 'test']

if __name__ == '__main__':

    # Charging model and tokenizer

    if model_path in {"mistralai/Mixtral-8x7B-Instruct-v0.1"}:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) 

        model = AutoModelForCausalLM.from_pretrained(
            "/data/tomas/models/" + model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    
    else:
        model = AutoModelForCausalLM.from_pretrained("/data/tomas/models/" + model_name, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained("/data/tomas/models/" + model_name)

    examples = data_train.sample(n=5, random_state=SEED)

    # Charging prompts
    with open("/data/tomas/hate/prompts/label_prompt_fs.txt", "r", encoding="utf-8") as f:
        label_prompt = f.read()

        examples_text = ""

        for index, row in examples.iterrows():

            examples_text += "Tweet: " + row["tweet_clean_lowercase"] + "\nClass: " + str(row["label"]) + "\n"

        label_prompt = label_prompt.replace("{examples}", examples_text)
    
    with open("/data/tomas/hate/prompts/target_prompt_fs.txt", "r", encoding="utf-8") as f:
        target_prompt = f.read()

        examples_text = ""

        for index, row in examples.iterrows():

            examples_text += "Tweet: " + row["tweet_clean_lowercase"] + "\nClass: " + str(row["target"]) + "\n"

        target_prompt = target_prompt.replace("{examples}", examples_text)

    with open("/data/tomas/hate/prompts/intensity_prompt_fs.txt", "r", encoding="utf-8") as f:
        intensity_prompt = f.read()

        examples_text = ""

        for index, row in examples.iterrows():

            examples_text += "Tweet: " + row["tweet_clean_lowercase"] + "\nClass: " + str(row["intensity"]).split("-")[0] + "\n"

        intensity_prompt = intensity_prompt.replace("{examples}", examples_text)

    with open("/data/tomas/hate/prompts/group_prompt_fs.txt", "r", encoding="utf-8") as f:
        group_prompt = f.read()

        label2id = {"safe": 0, "homophoby-related": 1, "misogyny-related": 2, "racism-related": 3, "fatphobia-related": 4, "transphoby-related": 5, "profession-related": 6, "disablism-related": 7, "aporophobia-related": 8}
        
        examples_text = ""

        for index, row in examples.iterrows():
            
            raw_classes = str(row["group"]).split(";")

            ids = [label2id[label.strip()] for label in raw_classes if label.strip() in label2id]

            if not ids:
                ids = [0]

            classes = " ,".join(map(str, ids))
            examples_text += "Tweet: " + row["tweet_clean_lowercase"] + "\nClasses: " + classes + "\n"

        group_prompt = group_prompt.replace("{examples}", examples_text)

    initial_prompts = [label_prompt, target_prompt, intensity_prompt, group_prompt]

    with open("/data/tomas/hate/results/fs/" + model_name  + "_responses.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["twitter_id", "label", "target", "intensity", "group"])

    # For each instance to evaluate
    for index, row in data_test.iterrows():

        responses = []

        for initial_prompt in initial_prompts:

            prompt = initial_prompt.replace("{tweet_text}", row["tweet_clean_lowercase"])

            if model_path in {"google/gemma-2-2b-it", "google/gemma-2-9b-it"}:

                conversation = [
                    {"role": "user", "content": prompt},
                ]

                inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt", return_dict=True).to("cuda")

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=32, use_cache=False)
                
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens = True).strip()

            elif model_path in {"google/gemma-3-1b-it"}:

                conversation = [
                    [
                        {
                            "role": "system", 
                            "content": [{"type": "text", "text": "You are a classification model that is really good at following instructions. Please follow the user's instructions as precisely as possible."},]
                        },
                        {
                            "role": "user", 
                            "content": [{"type": "text", "text": prompt},]
                        },
                    ]
                ]

                inputs = tokenizer.apply_chat_template(
                            conversation, 
                            tokenize=True, 
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt",
                    ).to("cuda")

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=32, use_cache=False)

                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            elif model_path in {"meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"}:

                conversation = [
                    {
                        "role": "system", 
                        "content": "You are a classification model that is really good at following instructions. Please follow the user's instructions as precisely as possible."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    },
                ]
                
                inputs = tokenizer.apply_chat_template(
                            conversation, 
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                    ).to("cuda")
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=32, use_cache=False)

                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            elif model_path in {"Qwen/Qwen2.5-7B-Instruct"}:

                conversation = [
                    {
                        "role": "system", 
                        "content": "You are a classification model that is really good at following instructions. Please follow the user's instructions as precisely as possible."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    },
                ]

                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer([text], return_tensors="pt").to("cuda")

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=32, use_cache=False)
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
                ]
                
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                del generated_ids

            elif model_path in {"Qwen/Qwen3-8B"}:

                conversation = [
                    {
                        "role": "system", 
                        "content": "You are a classification model that is really good at following instructions. Please follow the user's instructions as precisely as possible."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    },
                ]

                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )

                inputs = tokenizer(text, return_tensors="pt").to("cuda")

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=32, use_cache=False)
                
                generated_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
                
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)

                del generated_ids

            elif model_path in {"mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mixtral-8x7B-Instruct-v0.1"}:

                conversation = [
                    {
                        "role": "system", 
                        "content": "You are a classification model that is really good at following instructions. Please follow the user's instructions as precisely as possible."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    },
                ]

                inputs = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda")

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=32, use_cache=False)

                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            responses.append(response)
        
            del inputs
            del outputs
            torch.cuda.empty_cache()
            gc.collect()

        with open("/data/tomas/hate/results/fs/" + model_name  + "_responses.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([row["twitter_id"]] + responses)

        
    
    print("fin de ft " + model_name)
