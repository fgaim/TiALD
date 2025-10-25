# Baseline Models for the TiALD Benchmark

The training and inference code for the baseline approaches presented in the paper can be found here.

1. Single-task Fine-tuned Models
   - Entry script: `train_single_task.py`
2. Multi-task Fine-tuned Models
   - Entry script: `train_joint_tasks.py`
3. Evaluating LLMs in zero-shot and few-shot settings
   - Entry script: `run_llm_eval.py`
   - Data preparation: `prepare_data.py`

Each script has commandline options that can be shown by running `python <script-name> --help`.

## Install Dependencies

Install the dependencies for the training and evaluation scripts.

```sh
pip install -r requirements.txt
```

## Example Commands

Following is an example of how to run a single task training on the abusiveness task, followed by an evaluation that save prediction file.

```sh
lr="2e-5"
epochs=4
batch_size=16
max_length=256
output_dir "models"
hf_token=""

input_model_name="fgaim/tiroberta-base"
output_model_name="tiroberta-base"

python train_single_task.py \
    train abusiveness \
    --input_model_name ${input_model_name} \
    --output_dir ${output_dir} \
    --output_model_name ${output_model_name} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --epochs ${epochs} \
    --hf_token ${hf_token} \
    --report_to_wandb

input_model_name="${output_dir}/tiroberta-base_abusiveness"
python train_single_task.py \
    eval abusiveness \
    --input_model_name ${input_model_name} \
    --output_dir ${output_dir} \
    --output_model_name ${output_model_name} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --epochs ${epochs} \
    --hf_token ${hf_token}
```

Once the training and evaluation are complete, a predictions file named `predictions_test.json` will be save in the `<output_dir>/<output_model_name>` directory.
In the above example, it would be saved `models/tiroberta-base_abusiveness/predictions_test.json`.

The official TiALD metrics should be computed using the [`compute_tiald_metrics.py`](../compute_tiald_metrics.py).  
For the above example, we can use the below command to compute metrics, which will append the final scores to the prediction file itself:

```sh
python compute_tiald_metrics.py --append_metrics --prediction_file "models/tiroberta-base_abusiveness/predictions_test.json"
```

## LLM Evaluation

The LLM evaluation script (`run_llm_eval.py`) loads the TiALD dataset directly from Hugging Face Hub, so no data preparation is needed. Available options:

- `--model`: `gpt_4o`, `sonnet`, `llama`, `gemma`
- `--task`: `abusiveness`, `sentiment`, `topic`
- `--setting`: `zero`, `few`, `zero_title_desc`, `few_title_desc`
- `--run`: Run number (1, 2, etc.) for averaging results
- `--todo`: `gen` (generate predictions and evaluate) or `eval` (only evaluate existing predictions)

### Running Individual Experiments

To evaluate a specific LLM on a task:

```sh
# Example: Evaluate GPT-4o on sentiment task with zero-shot prompting
python run_llm_eval.py \
    --api_key YOUR_API_KEY \
    --model gpt_4o \
    --task sentiment \
    --setting zero \
    --run 1 \
    --todo gen
```

### Output Files

Predictions are stored efficiently:

- **One prediction file per model+setting+run**: `predictions_<model>_<setting>_<run>.json`
  - Contains all tasks: `abusiveness_predictions`, `sentiment_predictions`, `topic_predictions`
  - Script skips existing predictions when adding new tasks
- **One eval file per model+setting+task+run**: `eval_<model>_<setting>_<task>_<run>.json`
  - Contains accuracy, macro F1, per-class F1 scores
