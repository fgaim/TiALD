import argparse
import json
import os
import random
import re
import textwrap
import time
from datetime import datetime

import anthropic
import torch
from openai import OpenAI
from tiald_trainer.utils import compute_task_metrics
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(25)
device = torch.device("cuda")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model", type=str, choices=["gpt_4o", "sonnet", "llama", "gemma"])
    parser.add_argument("--setting", type=str, choices=["zero", "few", "zero_title_desc", "few_title_desc"])
    parser.add_argument("--todo", type=str, choices=["gen", "eval"])
    parser.add_argument("--max_token", type=int, default=100)
    parser.add_argument("--run", type=int, default=1)
    return parser.parse_args()


def read_jsonl_file(path):
    try:
        json_list = []
        with open(path, "r") as jsonl_file:
            for line in jsonl_file:
                data = json.loads(line)
                json_list.append(data)

        return json_list

    except Exception as e:
        print(f"Error: Fail to read jsonl file :: {e}")
        return None


def read_json_file(path):
    try:
        with open(path, "r") as json_file:
            json_data = json.load(json_file)

        return json_data

    except Exception as e:
        print(f"Error: Fail to read json file :: {e}")
        return {}


def get_model(args):
    if args.model == "gpt_4o":
        model_name = "gpt-4o-2024-08-06"
        model = OpenAI(api_key=args.api_key)
        tokenizer = None
    elif args.model == "sonnet":
        model_name = "claude-3-7-sonnet-20250219"
        model = anthropic.Anthropic(api_key=args.api_key)
        tokenizer = None
    elif args.model == "llama":
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif args.model == "gemma":
        model_name = "google/gemma-3-4b-it"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_info = [model_name, model, tokenizer]
    return model_info


def get_instruction(setting: str):

    if setting == "zero":
        instruction = textwrap.dedent(
            """
            Classify the following user comment written in Tigrinya as "Abusive" or "Not Abusive". Do not provide any explanation or additional text.
            Comment: {}
            """
        )

    elif setting == "few":
        instruction = textwrap.dedent(
            """
            Classify the following user comment written in Tigrinya as "Abusive" or "Not Abusive". Do not provide any explanation or additional text.
            Here are some examples:
            {}: {}
            {}: {}
            {}: {}
            {}: {}
            Comment: {}
            """
        )

    elif setting == "zero_title_desc":
        instruction = textwrap.dedent(
            """
            Below are the title and description of a YouTube video followed by a user comment written in Tigrinya.
            Classify the user comment either as "Abusive" or "Not Abusive". Do not provide any explanation or additional text, just the label.\n
            Title: {}
            Description: {}\n
            Comment: {}
            """
        )

    elif setting == "few_title_desc":
        instruction = textwrap.dedent(
            """
            Below are the title and description of a YouTube video followed by a user comment written in Tigrinya.
            Classify the user comment either as "Abusive" or "Not Abusive". Do not provide any explanation or additional text, just the label.\n
            Title: {}
            Description: {}\n\n
            Here are some examples comment-label pairs:
            {}: {}
            {}: {}
            {}: {}
            {}: {}
            Comment: {}
            """
        )

    return instruction.strip()


def generate_response(args, model_info, messages):

    cnt = 0
    response = None
    device = torch.device("cuda")

    model_name, model, tokenizer = model_info

    if args.model == "gpt_4o":
        while True:
            try:
                cnt += 1
                if cnt == 5:
                    break

                response = model.chat.completions.create(model=model_name, messages=messages)
                break

            except Exception as e:
                print("GPT Exception :: {}".format(e))
                time.sleep(10)

        if not response:
            print("Empty response from GPT")
            return None

        response_dict = response.to_dict()
        return response_dict["choices"][0]["message"]["content"]

    elif args.model == "sonnet":
        while True:
            try:
                cnt += 1
                if cnt == 5:
                    break

                response = model.messages.create(model=model_name, max_tokens=args.max_token, messages=messages)
                break

            except Exception as e:
                print("SONNET Exception :: {}".format(e))
                time.sleep(10)

        if not response:
            print("Empty response from SONNET")
            return None

        response_dict = response.to_dict()

        return response_dict["content"][0]["text"]

    elif args.model in ["llama", "gemma"]:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenized_prompt = tokenizer(prompt, return_tensors="pt")

        input_ids = tokenized_prompt["input_ids"].to(device)
        attention_mask = tokenized_prompt["attention_mask"].to(device)

        response = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_token,
            pad_token_id=tokenizer.eos_token_id,
        )

        decoded_response = tokenizer.decode(response[0][input_ids.shape[-1] :], skip_special_tokens=True)

        return decoded_response


def refine_response(response):

    refined_response = response

    if "Abusive\nNot Abusive\nAbusive\nNot Abusive" in refined_response:
        refined_response = refined_response.replace("Abusive\nNot Abusive\nAbusive\nNot Abusive", "")

    if "1. Abusive\n2. Not Abusive\n3. Abusive\n4. Not Abusive" in refined_response:
        refined_response = refined_response.replace("1. Abusive\n2. Not Abusive\n3. Abusive\n4. Not Abusive", "")

    refined_response = refined_response.replace("\n", "")

    return re.sub(r"[^a-zA-Z0-9\s]", "", refined_response).strip()


def gen_LLM(args):

    output_dir_name = "frontier-llms" if args.model in ["gpt_4o", "sonnet"] else "open-source-llms"
    prediction_json_path = "./model-predictions/{}/predictions_{}_{}_{}.json".format(
        output_dir_name, args.model, args.setting, str(args.run)
    )

    if os.path.isfile(prediction_json_path):
        print(f"Error: Prediction file already exists! {prediction_json_path}")
        return

    train_data = read_jsonl_file("./data/TiALD-train.jsonl")
    test_data = read_jsonl_file("./data/TiALD-test.jsonl")

    if train_data is None or test_data is None:
        print("Error: Fail to read data")
        return

    instruction_template = get_instruction(args.setting)
    abusive_list = [data["comment_clean"] for data in train_data if data["abusiveness"] == "Abusive"]
    not_abusive_list = [data["comment_clean"] for data in train_data if data["abusiveness"] == "Not Abusive"]

    model_info = get_model(args)
    model_predictions = {
        "config": {
            "model_name": args.model,
            "prompt_type": args.setting,
            "test_date": datetime.today().strftime("%Y%m%d"),
        },
        "abusiveness_predictions": {},
        "abusiveness_predictions_responses": {},
        "topic_predictions": {},
        "sentiment_predictions": {},
    }

    for data in tqdm(test_data):
        if args.setting == "zero":
            instruction = instruction_template.format(data["comment_clean"])

        elif args.setting == "few":
            rand_abusive = random.sample(abusive_list, 2)
            rand_not_abusive = random.sample(not_abusive_list, 2)
            instruction = instruction_template.format(
                rand_abusive[0],
                "Abusive",
                rand_not_abusive[0],
                "Not Abusive",
                rand_abusive[1],
                "Abusive",
                rand_not_abusive[1],
                "Not Abusive",
                data["comment_clean"],
            )

        elif args.setting == "zero_title_desc":
            instruction = instruction_template.format(
                data["video_title"],
                data["video_description"],
                data["comment_clean"],
            )

        elif args.setting == "few_title_desc":
            rand_abusive = random.sample(abusive_list, 2)
            rand_not_abusive = random.sample(not_abusive_list, 2)
            instruction = instruction_template.format(
                data["video_title"],
                data["video_description"],
                rand_abusive[0],
                "Abusive",
                rand_not_abusive[0],
                "Not Abusive",
                rand_abusive[1],
                "Abusive",
                rand_not_abusive[1],
                "Not Abusive",
                data["comment_clean"],
            )
        else:
            raise ValueError(f"Invalid setting: {args.setting}")

        messages = [{"role": "user", "content": instruction}]
        response = generate_response(args, model_info, messages)
        model_predictions["abusiveness_predictions_responses"][data["comment_id"]] = response

    with open(prediction_json_path, "w") as out:
        json.dump(model_predictions, out, indent=2, sort_keys=False, ensure_ascii=False)
    print(f"Model predictions saved to {prediction_json_path}")


def eval_LLM(args):

    abuse_labels = ["Abusive", "Not Abusive"]
    output_dir_name = "frontier-llms" if args.model in ["gpt_4o", "sonnet"] else "open-source-llms"
    preds_file = f"./model-predictions/{output_dir_name}/predictions_{args.model}_{args.setting}_{args.run}.json"

    test_data = read_jsonl_file("./data/TiALD-test.jsonl")
    if test_data is None:
        print("Error: Fail to read test data")
        return

    predicted_data = read_json_file(preds_file)
    if predicted_data is None:
        print("Error: Fail to read json file")
        return

    # refine generated responses
    for cid in predicted_data["abusiveness_predictions_responses"].keys():
        response = predicted_data["abusiveness_predictions_responses"][cid]
        refined_response = refine_response(response)

        if refined_response not in abuse_labels:
            refined_response = "Invalid"

        predicted_data["abusiveness_predictions"][cid] = refined_response

    with open(preds_file, "w") as out:
        json.dump(predicted_data, out, indent=2, sort_keys=False, ensure_ascii=False)

    # evaluation
    golden_answers = [data["abusiveness"] for data in test_data]
    predicted_answers = [predicted_data["abusiveness_predictions"][data["comment_id"]] for data in test_data]
    result = compute_task_metrics(golden_answers, predicted_answers, abuse_labels)

    eval_file = f"./model-predictions/{output_dir_name}/eval_{args.model}_{args.setting}_{args.run}.json"
    with open(eval_file, "w") as fout:
        json.dump(result, fout, indent=2, sort_keys=False, ensure_ascii=False)
    print(f"Metrics saved to {eval_file}")


if __name__ == "__main__":

    args = parse_args()

    if args.todo == "gen":
        gen_LLM(args)
        eval_LLM(args)

    elif args.todo == "eval":
        eval_LLM(args)
