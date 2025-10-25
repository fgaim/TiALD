import argparse
import json
import os
import random
import re
import textwrap
import time
from datetime import datetime
from typing import Any

import anthropic
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiald_trainer.utils import compute_task_metrics, TASK_INFO

random.seed(25)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK_KEYS = tuple(TASK_INFO.keys())
SETTING_KEYS = ["zero", "few", "zero_title", "few_title", "zero_title_desc", "few_title_desc"]
MODEL_KEYS = ["gpt_4o", "sonnet", "llama", "gemma"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=TASK_KEYS, help=f"Task to evaluate one of {TASK_KEYS}")
    parser.add_argument("--model", type=str, choices=MODEL_KEYS, help=f"Model to evaluate one of {MODEL_KEYS}")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--setting", type=str, choices=SETTING_KEYS, help=f"Setting to evaluate one of {SETTING_KEYS}")
    parser.add_argument("--todo", type=str, choices=["gen", "eval"])
    parser.add_argument("--max_token", type=int, default=100)
    parser.add_argument("--run", type=int, default=1)
    return parser.parse_args()


def load_tiald_dataset():
    """Load TiALD dataset from Hugging Face Hub."""
    try:
        print("Loading TiALD dataset from Hugging Face Hub...")
        dataset = load_dataset("fgaim/tigrinya-abusive-language-detection")
        print("✓ Dataset loaded successfully")
        return dataset
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nMake sure you have internet connection and 'datasets' package installed:")
        return None


def read_json_file(path):
    try:
        with open(path, "r") as json_file:
            json_data = json.load(json_file)

        return json_data
    except Exception as e:
        print(f"Error: Fail to read json file :: {e}")
        return {}


def get_task_labels(task) -> tuple[str]:
    """Get the list of valid labels for a given task."""
    if task == "abusiveness":
        return ("Not Abusive", "Abusive")  # Order matters for label refinement

    return TASK_INFO[task]["label_list"]  # sentiment, topic


def get_task_field(task) -> str:
    """Get the field name in the dataset for a given task."""
    return TASK_INFO[task.lower().strip()]["label_column"]


def load_model_info(args) -> tuple[str, Any, Any]:
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


def get_instruction(setting: str, task: str):
    """Generate instruction template based on setting and task."""

    # Task-specific classification instructions
    task_instructions = {
        "abusiveness": 'Classify the following user comment written in Tigrinya as "Abusive" or "Not Abusive".',
        "sentiment": 'Classify the sentiment of the following user comment written in Tigrinya as one of: "Positive", "Neutral", "Negative", or "Mixed".',
        "topic": 'Classify the topic of the following user comment written in Tigrinya as one of: "Political", "Racial", "Sexist", "Religious", or "Other".',
    }

    task_instruction = task_instructions[task]

    if setting == "zero":
        instruction = textwrap.dedent(
            f"""
            {task_instruction} Do not provide any explanation or additional text, just the label.\n
            Comment: {{user_comment}}
            """
        )

    elif setting == "few":
        instruction = textwrap.dedent(
            f"""
            {task_instruction} Do not provide any explanation or additional text, just the label.\n
            Here are some examples of comment-label pairs for reference:
            {{few_shot_examples}}\n
            Now, return the label for the following comment.
            Comment: {{user_comment}}
            """
        )

    elif setting == "zero_title":
        instruction = textwrap.dedent(
            f"""
            Below is the title of a YouTube video followed by a user comment written in Tigrinya.
            {task_instruction} Do not provide any explanation or additional text, just the label.\n
            Video Title: {{video_title}}
            Comment: {{user_comment}}
            """
        )

    elif setting == "few_title":
        instruction = textwrap.dedent(
            f"""
            Below is the title of a YouTube video followed by a user comment written in Tigrinya.
            {task_instruction} Do not provide any explanation or additional text, just the label.\n
            Here are some examples of comment-label pairs for reference:
            {{few_shot_examples}}\n
            Now, return the label for the following comment.
            Video Title: {{video_title}}
            Comment: {{user_comment}}
            """
        )

    elif setting == "zero_title_desc":
        instruction = textwrap.dedent(
            f"""
            Below are the title and description of a YouTube video followed by a user comment written in Tigrinya.
            {task_instruction} Do not provide any explanation or additional text, just the label.\n
            Video Title: {{video_title}}
            Video Description:
            ```
            {{video_description}}
            ```\n
            Comment: {{user_comment}}
            """
        )

    elif setting == "few_title_desc":
        instruction = textwrap.dedent(
            f"""
            Below are the title and description of a YouTube video followed by a user comment written in Tigrinya.
            {task_instruction} Do not provide any explanation or additional text, just the label.\n
            Here are some examples of comment-label pairs for reference:
            {{few_shot_examples}}\n
            Now, return the label for the following comment.
            Video Title: {{video_title}}
            Video Description:
            ```
            {{video_description}}
            ```\n
            Comment: {{user_comment}}
            """
        )

    return instruction.strip()


def generate_response(args, model_info, messages):
    cnt = 0
    response = None

    model_name, model, tokenizer = model_info

    if isinstance(model, OpenAI):
        while True:
            try:
                cnt += 1
                if cnt >= 5:
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

    elif isinstance(model, anthropic.Anthropic):
        while True:
            try:
                cnt += 1
                if cnt >= 5:
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


def refine_response(response: str, task: str) -> str:
    """Refine the raw model response to extract the classification label.

    Returns "Invalid" if:
        - Response contains multiple different labels (ambiguous)
        - No valid label can be extracted
        - Response is empty
    """
    if not response or not response.strip():
        return "Invalid"

    response = response.strip()
    valid_labels = get_task_labels(task)

    # Count occurrences of each label in the raw response
    label_counts = {}
    for label in valid_labels:
        # Case-insensitive count of how many times this label appears
        pattern = r"\b" + re.escape(label) + r"\b"
        matches = re.findall(pattern, response, flags=re.IGNORECASE)
        if matches:
            label_counts[label] = len(matches)

    # If multiple different labels appear, it's ambiguous -> Invalid
    if len(label_counts) > 1:
        # Examples like "Abusive\nNot Abusive\nAbusive\nNot Abusive" should be Invalid
        total_label_occurrences = sum(label_counts.values())
        if total_label_occurrences > 2:  # More than 2 label mentions is suspicious
            return "Invalid"

    # Try exact match first (case-insensitive)
    response_clean = response.lower().strip()
    for label in valid_labels:
        if label.lower() == response_clean:
            return label

    # Try to find label as isolated word
    response_words = re.sub(r"[^a-zA-Z0-9\s]", " ", response).lower().split()
    for label in valid_labels:
        label_words = label.lower().split()
        # Check if all words of the label appear consecutively
        if len(label_words) == 1 and label_words[0] in response_words:
            return label
        elif len(label_words) > 1:
            label_str = " ".join(label_words)
            if label_str in " ".join(response_words):
                return label

    # If exactly one label was found in counts, return it
    if len(label_counts) == 1:
        return list(label_counts.keys())[0]

    # No valid label found
    return "Invalid"


def gen_LLM(args):
    """Generate LLM predictions for a given task, model, setting, and run."""

    print("=" * 80)
    print(f"Generating {args.task} predictions for {args.model} with {args.setting} setting and run {args.run}")
    print("-" * 80)

    output_dir_name = "frontier-llms" if args.model in ["gpt_4o", "sonnet"] else "open-source-llms"
    output_dir_path = f"./model-predictions/{output_dir_name}"
    os.makedirs(output_dir_path, exist_ok=True)

    prediction_json_path = f"{output_dir_path}/predictions_{args.model}_{args.setting}_{args.run}.json"

    dataset = load_tiald_dataset()
    if dataset is None:
        print("Error: Failed to load dataset")
        return

    if os.path.isfile(prediction_json_path):
        print(f"Loading existing predictions from {prediction_json_path}")
        model_predictions = read_json_file(prediction_json_path)
        if model_predictions is None:
            print("Error: Could not load existing file")
            return
        for task in TASK_KEYS:
            model_predictions[f"{task}_predictions"] = model_predictions.get(f"{task}_predictions", {})
            model_predictions[f"{task}_predictions_responses"] = model_predictions.get(f"{task}_predictions_responses", {})
    else:
        print(f"Creating new prediction file: {prediction_json_path}")
        model_predictions = {
            "config": {
                "model_name": args.model,
                "prompt_type": args.setting,
                "test_date": datetime.today().strftime("%Y%m%d"),
            },
            "abusiveness_predictions": {},
            "abusiveness_predictions_responses": {},
            "topic_predictions": {},
            "topic_predictions_responses": {},
            "sentiment_predictions": {},
            "sentiment_predictions_responses": {},
        }

    test_data = dataset["test"]
    task_field = get_task_field(args.task)
    task_labels = get_task_labels(task_field)
    label_examples = {}
    if args.setting.startswith("few"):
        # Organize train set by label for few-shot sampling
        train_data = dataset["train"]
        for label in task_labels:
            label_examples[label] = [data["comment_clean"] for data in train_data if data[task_field] == label]

    model_info = load_model_info(args)
    response_key = f"{task_field}_predictions_responses"
    instruction_template = get_instruction(args.setting, task_field)

    existing_responses_count = 0

    progress = 0
    for sample in tqdm(test_data, desc=f"Generating {task_field} predictions"):
        # Skip if prediction already exists for this sample
        _response = model_predictions.get(response_key, {}).get(sample["comment_id"])
        if _response and _response.strip():
            existing_responses_count += 1
            continue

        if args.setting.startswith("few"):
            examples = []
            for label in task_labels:
                if len(label_examples[label]) >= 2:
                    sampled = random.sample(label_examples[label], 2)
                    examples.append((sampled[0], label))
                    examples.append((sampled[1], label))
                elif len(label_examples[label]) == 1:
                    examples.append((label_examples[label][0], label))
            random.shuffle(examples)  # Shuffle to avoid order bias
            # Format examples as "Comment: Label" pairs separated by newlines
            examples_str = "\n".join([f"- {comment}: {label}" for comment, label in examples])

        if args.setting == "zero":
            instruction = instruction_template.format(user_comment=sample["comment_clean"])

        elif args.setting == "few":
            instruction = instruction_template.format(
                few_shot_examples=examples_str,
                user_comment=sample["comment_clean"],
            )

        elif args.setting == "zero_title":
            instruction = instruction_template.format(
                video_title=sample["video_title"],
                user_comment=sample["comment_clean"],
            )

        elif args.setting == "few_title":
            instruction = instruction_template.format(
                few_shot_examples=examples_str,
                video_title=sample["video_title"],
                user_comment=sample["comment_clean"],
            )

        elif args.setting == "zero_title_desc":
            instruction = instruction_template.format(
                video_title=sample["video_title"],
                video_description=sample["video_description"],
                user_comment=sample["comment_clean"],
            )

        elif args.setting == "few_title_desc":
            instruction = instruction_template.format(
                few_shot_examples=examples_str,
                video_title=sample["video_title"],
                video_description=sample["video_description"],
                user_comment=sample["comment_clean"],
            )
        else:
            raise ValueError(f"Invalid setting: {args.setting}")

        messages = [{"role": "user", "content": instruction}]
        response = generate_response(args, model_info, messages)
        model_predictions[response_key][sample["comment_id"]] = response

        progress += 1
        if progress % 10 == 0:
            print(f"Instruction:\n```\n{instruction}\n```")
            with open(prediction_json_path, "w") as out:
                json.dump(model_predictions, out, indent=2, sort_keys=False, ensure_ascii=False)
            print(f"Intermediate predictions saved to {prediction_json_path}")

    if existing_responses_count:
        total_count = len(test_data)
        if existing_responses_count == total_count:
            print(f"All {total_count} predictions for {task_field} already exist. Skipped generation.")
            return
        print(
            f"Found {existing_responses_count}/{total_count} existing predictions. "
            f"Generating remaining {total_count - existing_responses_count}."
        )

    with open(prediction_json_path, "w") as out:
        json.dump(model_predictions, out, indent=2, sort_keys=False, ensure_ascii=False)
    print(f"Model predictions saved to {prediction_json_path}")


def eval_LLM(args):
    """Compute metrics for existing LLM predictions."""

    print("=" * 80)
    print(f"Evaluating {args.task} predictions for {args.model} with {args.setting} setting and run {args.run}")
    print("-" * 80)

    task_field = get_task_field(args.task)
    task_labels = get_task_labels(task_field)
    output_dir_name = "frontier-llms" if args.model in ["gpt_4o", "sonnet"] else "open-source-llms"
    preds_file = f"./model-predictions/{output_dir_name}/predictions_{args.model}_{args.setting}_{args.run}.json"

    dataset = load_tiald_dataset()
    if dataset is None:
        print("Error: Failed to load test data")
        return

    predicted_data = read_json_file(preds_file)
    if predicted_data is None:
        print(f"Error: Prediction file not found: {preds_file}")
        return

    prediction_key = f"{task_field}_predictions"
    response_key = f"{task_field}_predictions_responses"

    if response_key not in predicted_data or len(predicted_data[response_key]) == 0:
        print(f"Error: No {task_field} predictions found in {preds_file}")
        print("Please run generation first: `--todo gen`")
        return

    # refine generated responses
    for cid in predicted_data[response_key].keys():
        response = predicted_data[response_key][cid]
        refined_response = refine_response(response=response, task=task_field) if response else ""

        if refined_response not in task_labels:
            refined_response = "Invalid"

        predicted_data[prediction_key][cid] = refined_response

    test_data = dataset["test"]
    golden_answers = [data[task_field] for data in test_data]
    predicted_answers = [predicted_data[prediction_key][data["comment_id"]] for data in test_data]
    eval_metrics = compute_task_metrics(golden_answers, predicted_answers)  # task_labels is not needed
    predicted_data[f"{task_field}_metrics"] = eval_metrics
    with open(preds_file, "w") as out:
        json.dump(predicted_data, out, indent=2, sort_keys=False, ensure_ascii=False)
    print(f"Predictions and Metrics saved to {preds_file}")

    print(f"\n{task_field.capitalize()} Task - {args.model} - {args.setting} - Run {args.run}")
    print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"Macro F1: {eval_metrics['macro_f1']:.4f}")
    print("Per-class F1 scores:")
    for label in task_labels:
        if label in eval_metrics["per_class_metrics"]:
            print(f"  - {label}: {eval_metrics['per_class_metrics'][label]['f1-score']:.4f}")


if __name__ == "__main__":
    args = parse_args()
    if args.todo == "gen":
        gen_LLM(args)
        eval_LLM(args)
    elif args.todo == "eval":
        eval_LLM(args)
