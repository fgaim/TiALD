# Baseline Models for the TiALD Benchmark

The training and inference code for the baseline approaches presented in the paper can be found here.

1. Single-task Fine-tuned Models
   - Entry script: `train_single_task.py`
2. Multi-taskk Fine-tuned Models
   - Entry script: `train_joint_tasks.py`
3. Evaluating LLMs in zero-shot and few-shot settings
   - Entry script: `run_llm_eval.py`

Each script has commandline options that can be show by running `python <script-name> --help`.

## Install Dependencies

Install the dependencies for the training and evaluation scripts.

```
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
