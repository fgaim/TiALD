# Tigrinya Abusive Language Detection (TiALD) Dataset

**Tigrinya Abusive Language Dataset (TiALD)** is a large-scale, multi-task benchmark dataset for abusive language detection in the Tigrinya language. It consists of **13,717 YouTube comments** annotated for **abusiveness**, **sentiment**, and **topic** tasks. The dataset includes comments written in both the **Ge’ez script** and prevalent non-standard Latin-based **transliterations** to mirror real-world usage.

The dataset also includes contextual metadata such as video titles and VLM-generated and LLM-enhanced descriptions of the corresponding video content, enabling context-aware modeling.

⚠️ **The dataset contains explicit, obscene, and hateful language. It should be used for research purposes only.** ⚠️

This work accompanies the paper ["A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings"](https://arxiv.org/abs/2505.12116), accepted at the **NeurIPS 2025** conference in the Datasets & Benchmarks Track, San Diego, December (2-7), 2025.

**Outline:**

- [Dataset Overview](#dataset-overview)
  - [Tasks and Annotation Schema](#tasks-and-annotation-schema)
  - [How to access the TiALD Dataset](#how-to-access-the-tiald-dataset)
- [Baseline Models and Results](#baseline-models-and-results)
  - [Trained Baseline Models](#trained-baseline-models)
  - [Code for Baseline Models](#code-for-baseline-models)
  - [1. Main Results: Performance of Fine-tuned and Prompted Models](#1-main-results-performance-of-fine-tuned-and-prompted-models)
  - [2. Performance of Models with Video Title as Context](#2-performance-of-models-with-video-title-as-context)
  - [3. Performance of LLMs on Abusiveness Detection with Cross-Modality Context](#3-performance-of-llms-on-abusiveness-detection-with-cross-modality-context)
  - [Baseline Models Prediction Files](#baseline-models-prediction-files)
- [Dataset Details](#dataset-details)
  - [Dataset Statistics](#dataset-statistics)
  - [Dataset Features](#dataset-features)
  - [Inter-Annotator Agreement (IAA)](#inter-annotator-agreement-iaa)
  - [Croissant Metadata for TiALD Dataset](#croissant-metadata-for-tiald-dataset)
- [Intended Usage of TiALD Dataset](#intended-usage-of-tiald-dataset)
  - [Ethical Considerations](#ethical-considerations)
- [Evaluation and Computing Metrics](#evaluation-and-computing-metrics)
  - [Model Predictions File Format](#model-predictions-file-format)
  - [Computing Metrics](#computing-metrics)
- [Citation](#citation)
- [License](#license)

## Dataset Overview

- **Data Source**: YouTube comments from 51 popular channels in the Tigrinya-speaking community.
- **Scope**: 13,717 human-annotated comments from 7,373 videos with over 1.2 billion cumulative views at the time of collection.
- **Sampling**: Comments selected using an embedding-based semantic expansion strategy from an initial pool of ~4.1 million comments across ~34.5k videos.
- For data construction methodology, baseline results, and task formulation, see the associated paper.

### Tasks and Annotation Schema

TiALD supports multi-task modeling of three complementary tasks abusiveness, sentiment, and topic classification, which in turn has the following classes:

1. **Abusiveness**: Binary (`Abusive`, `Not Abusive`)
2. **Sentiment**: 4-way (`Positive`, `Neutral`, `Negative`, `Mixed`)
3. **Topic**: 5-way (`Political`, `Racial`, `Sexist`, `Religious`, `Other`)

A schematic overview of the dataset tasks and classes is shown below:

<div style="display: flex; justify-content: space-between; gap: 20px;">
  <img title="TiALD Annotation Schema" src="assets/tiald-schema.jpg" height=350 />
  <img title="TiALD Class Distribution" src="assets/tiald-class-dist.png" height=350 />
</div>

### How to Access the TiALD Dataset

A stable version of TiALD dataset is made available on 🤗 Hugging Face Hub.

You can head over to: <https://huggingface.co/datasets/fgaim/tigrinya-abusive-language-detection>

Or pull it from anywhere as follows:

```python
from datasets import load_dataset

dataset = load_dataset("fgaim/tigrinya-abusive-language-detection")
print(dataset["validation"][5])  # Inspect a sample
```

## Baseline Models and Results

### Trained Baseline Models

Some strong performing trained models trained on TiALD can be found on Hugging Face Hub:

- [Single-task Abusiveness Detection with TiRoBERTa](https://huggingface.co/fgaim/tiroberta-abusiveness-detection)
- [Multi-task TiALD with TiRoBERTa](https://huggingface.co/fgaim/tiroberta-tiald-multi-task)

### Code for Baseline Models

The training and inference code for the three baseline approaches discussed in the paper can be found in the [`baselines`](./baselines/) directory.

The following tables show the performances of the baseline models reported in the paper:

### 1. Main Results: Performance of Fine-tuned and Prompted Models

| Model | Abusiveness | Sentiment | Topic | TiALD Score |
|:---|---:|---:|---:|---:|
| **Fine-tuned Single-task Models** | | | | |
| TiELECTRA-small | 82.33  | 42.39  | 26.90  | 50.54  |
| TiRoBERTa-base | ***86.67***  | 52.82  | *54.23*  | *64.57*  |
| AfriBERTa-base | 83.42  | 50.81  | 53.20  | 62.48  |
| Afro-XLMR-Large-76L | 85.20  | ***54.94***  | 51.42  | 63.86  |
| XLM-RoBERTa-base | 81.08  | 30.17  | 43.97  | 51.74  |
| **Fine-tuned Multi-task Models** | | | | |
| TiELECTRA-small | 84.21  | 43.44  | 29.27  | 52.30  |
| TiRoBERTa-base | *86.11*  | 53.41  | ***54.91***  | ***64.81***  |
| AfriBERTa-base | 83.66  | 50.19  | 53.49  | 62.45  |
| Afro-XLMR-Large-76L | 85.44  | *54.50*  | 52.46  | 64.13  |
| XLM-RoBERTa-base | 79.87  | 45.40  | 35.50  | 53.59  |
| **Zero-shot Prompted LLMs** | | | | |
| GPT-4o | *71.05*  | 20.55  | 26.25  | 39.28  |
| Claude Sonnet 3.7 | 59.20  | 22.64  | 25.25  | 35.70  |
| Gemma-3 4B | 59.35  | *29.47*  | *35.24*  | *41.35*  |
| LLaMA-3.2 3B | 49.98  | 25.30  | 16.55  | 30.61  |
| **Few-shot Prompted LLMs** | | | | |
| GPT-4o | 72.06  | 21.88  | 27.56  | 40.50  |
| Claude Sonnet 3.7 | *79.31*  | 23.39  | 27.92  | *43.54*  |
| Gemma-3 4B | 58.37  | *30.46*  | *39.49*  | 42.78  |
| LLaMA-3.2 3B | 45.65  | 19.94  | 21.68  | 29.09  |

*Performance of fine-tuned encoder models (single and multi-task) and prompted generative LLMs (zero-shot and few-shot) evaluated on user comments across all three tasks. The TiALD Score is the average macro F1 across the three tasks. Overall task-level best scores are in **bold**; category-best scores are *italicized*.*

### 2. Performance of Models with Video Title as Context

| Model | Abusiveness | Sentiment | Topic | TiALD Score |
|:---|---:|---:|---:|---:|
| **Fine-tuned Single-task Models** | | | | |
| TiELECTRA-small | 81.67  | 39.40  | 27.81  | 49.62  |
| TiRoBERTa-base | ***86.17***  | ***54.97***  | ***54.55***  | ***65.23***  |
| AfriBERTa-base | 82.44  | 51.33  | 52.10  | 61.96  |
| Afro-XLMR-Large-76L | 84.20  | 52.64  | 54.11  | 63.65  |
| XLM-RoBERTa-base | 75.09  | 43.47  | 41.60  | 53.39  |
| **Zero-shot Prompted LLMs** | | | | |
| GPT-4o | *75.59*  | 41.03  | *55.52*  | *57.38*  |
| Claude Sonnet 3.7 | 67.64  | *44.39*  | 50.10  | 54.05  |
| Gemma-3 4B | 58.41  | 29.27  | 34.44  | 40.71  |
| LLaMA-3.2 3B | 44.13  | 21.85  | 15.91  | 27.30  |
| **Few-shot Prompted LLMs** | | | | |
| GPT-4o | 75.89  | 45.50  | 58.59  | 59.99  |
| Claude Sonnet 3.7 | *80.29*  | *48.01*  | *59.45*  | *62.58*  |
| Gemma-3 4B | 59.39  | 30.43  | 39.60  | 43.14  |
| LLaMA-3.2 3B | 48.29  | 20.19  | 20.20  | 29.56  |

*Performance of models with video title as context. Fine-tuned models were trained on concatenation of user comment and video title. LLMs were prompted with both comment and video title. Overall task-level best scores are in **bold**; category-best scores are *italicized*.*

### 3. Performance of LLMs on Abusiveness Detection with Cross-Modality Context

| Model | Comment Only | | Video Title + Comment | |
|:---|---:|---:|---:|---:|
| | **Zero-shot** | **Few-shot** | **Zero-shot** | **Few-shot** |
| **Closed Frontier Models** | | | | |
| GPT-4o | ***71.05***  | 72.06  | ***75.59***  | 75.89 |
| Claude Sonnet 3.7 | 59.20  | ***79.31***  | 67.64  | ***80.29*** |
| **Open-weight Models** | | | | |
| Gemma-3 4B | *59.35*  | *58.37*  | *58.41*  | *59.39* |
| LLaMA-3.2 3B | 49.98  | 45.65  | 44.13  | 48.29 |

*Performance of LLMs on Abusiveness Detection with Cross-Modality Contextual Information: user `comment` augmented with `video_title` and auto-generated `video_description`. Best scores for each prompting approach are in **bold**; highest scores within model category are *italicized*.*

*†LLaMA-3.2 3B produced invalid responses for over 61% of queries in both few-shot settings, mainly due to its limited Tigrinya text understanding.*

### Baseline Models Prediction Files

The final prediction files from baselines models reported in the paper can be found under the [`model-predictions`](./model-predictions/) folder.

## Dataset Details

### Dataset Statistics

A table summarizing the dataset splits and distributions of samples:

|   Split    | Samples | Abusive | Not Abusive | Political | Racial | Sexist | Religious | Other Topics | Positive | Neutral | Negative | Mixed |
|:----------:|:-------:|:-------:|:-----------:|:---------:|:------:|:------:|:---------:|:-------------:|:--------:|:-------:|:--------:|:-----:|
| Train      | 12,317  |  6,980  |    5,337    |   4,037   |  633   |  564   |    244    |     6,839     |  2,433   |  1,671  |   6,907  | 1,306  |
| Test       |   900   |   450   |     450     |    279    |  113   |   78   |    157    |      273      |   226    |   129   |   474    |  71   |
| Dev        |   500   |   250   |     250     |    159    |   23   |   21   |     11    |      286      |   108    |    71   |   252    |  69   |
| **Total**  | 13,717  |  7,680  |    6,037    |   4,475   |  769   |  663   |    412    |     7,398     |  2,767   |  1,871  |   7,633  | 1,446  |

### Dataset Features

Below is a complete list of features in the dataset, grouped by type:

| **Feature**               | **Type**    | **Description**                                                |
|---------------------------|-------------|----------------------------------------------------------------|
| `sample_id`               | Integer     | Unique identifier for the sample.                              |
| **Comment Information**   |             |                                                                |
| `comment_id`              | String      | YouTube comment identifier.                                    |
| `comment_original`        | String      | Original unprocessed comment text.                             |
| `comment_clean`           | String      | Cleaned version of the comment for modeling purposes.          |
| `comment_script`          | Categorical | Writing system of the comment: `geez`, `latin`, or `mixed`.    |
| `comment_publish_date`    | String      | Year and month when the comment was published, eg., 2021.11.   |
| **Comment Annotations**   |             |                                                                |
| `abusiveness`             | Categorical | Whether the comment is `Abusive` or `Not Abusive`.             |
| `topic`                   | Categorical | One of: `Political`, `Racial`, `Religious`, `Sexist`, or `Other`. |
| `sentiment`               | Categorical | One of: `Positive`, `Neutral`, `Negative`, or `Mixed`.         |
| `annotator_id`            | String      | Unique identifier of the annotator.                            |
| **Video Information**     |             |                                                                |
| `video_id`                | String      | YouTube video identifier.                                      |
| `video_title`             | String      | Title of the YouTube video.                                    |
| `video_publish_year`      | Integer     | Year the video was published, eg., 2022.                       |
| `video_num_views`         | Integer     | Number of views at the time of data collection.                |
| `video_description`       | String      | **Generated** description of video content using a vision-language model and refined by an LLM. |
| **Channel Information**   |             |                                                                |
| `channel_id`              | String      | Identifier for the YouTube channel.                            |
| `channel_name`            | String      | Name of the YouTube channel.                                   |

### Inter-Annotator Agreement (IAA)

To assess annotation quality, a subset of 900 comments was double-annotated, exact agreement across all tasks in 546 examples and partial disagreement 354 examples.

**Aggregate IAA Scores**:

| Task | Cohen's Kappa | Remark |
|:---|:---|:---|
|Abusiveness detection | 0.758 | Substantial agreement |
|Sentiment analysis    | 0.649 | Substantial agreement |
|Topic classification  | 0.603 | Moderate agreement |

**Gold label**: Expert adjudication was used to determine the final label of the test set, enabling a gold-standard evaluation.

### Croissant Metadata for TiALD Dataset

Croissant is an open, standardized metadata format designed to describe machine learning (ML) datasets. Its primary goal is to make datasets easily discoverable, interoperable, and usable across various ML tools, frameworks, and repositories without changing the underlying data files themselves.

The Croissant metadata for TiALD dataset can be found at [TiALD.Croissant.json](./data/TiALD.Croissant.json).

## Intended Usage of TiALD Dataset

The dataset is solely designed to support:

- Research in abusive language detection in low-resource languages
- Context-aware abusiveness, sentiment, and topic modeling
- Multi-task and transfer learning with digraphic scripts
- Evaluation of multilingual and fine-tuned language models

Researchers and developers should avoid using this dataset for direct moderation or enforcement tasks without human oversight.

### Ethical Considerations

- **Sensitive content**: Contains toxic and offensive language. Use for research purposes only.
- **Cultural sensitivity**: Abuse is context-dependent; annotations were made by native speakers to account for cultural nuance.
- **Bias mitigation**: Data sampling and annotation were carefully designed to minimize reinforcement of stereotypes.
- **Privacy**: All the source content for the dataset is publicly available on YouTube.
- **Respect for expression**: The dataset should not be used for automated censorship without human review.

This research received IRB approval (Ref: KH2022-133) from Korea Advanced Institute of Science and Technology (KAIST) and followed all ethical data collection and annotation practices, including informed consent of annotators.

## Evaluation and Computing Metrics

### Model Predictions File Format

Before computing metrics, you need to save models predictions for one or more of the three tasks in TiALD into a JSON file.

For consistency, we recommend saving the predictions into a file with the following format:

```json
{
    "config": {
        "model_name": "<unique model name>",
        "test_date": "<yyyymmdd>",
        "<custom-field>": "<e.g., model type, hyperparams>"
    },
    "abusiveness_predictions": {
        "<cid>": "<Abusive | Not Abusive>"
    },
    "topic_predictions": {
        "<cid>": "<Political | Religious | Sexist | Racial | Other>"
    },
    "sentiment_predictions": {
        "<cid>": "<Positive | Negative | Neutral | Mixed>"
    }
}
```

### Computing Metrics

Given an exising predictions file for the samples in TiALD test set, the `compute_tiald_metrics.py` script can be used to compute all metrics discussed in the paper (task-level and pre-class).

Install dependencies:

```sh
pip install scikit-learn datasets
```

Then run the script as follows:

```sh
python compute_tiald_metrics.py \
  --prediction_file <path-to-model-predictions.json> \
  [--output_file <output-file-to-save-results.json>]
  [--append_metrics <append metrics to the prediction file>]
```

The script automatically loads the TiALD dataset and computes the following metrics:

- Accuracy for each task
- Macro F1 scores for each task
- Per-class precision, recall, and F1 scores

The summary of results is logged to the terminal and can optionally be saved to a detailed JSON file using the `--output_file` flag.
The aggregate `TiALD Score` reported in the paper is an arthmetic mean of the task-level macro F1 scores.

## Citation

If you use `TiALD` in your work, please cite:

```bibtex
@misc{gaim-etal-2025-tiald-benchmark,
  title         = {A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings},
  author        = {Fitsum Gaim and Hoyun Song and Huije Lee and Changgeon Ko and Eui Jun Hwang and Jong C. Park},
  year          = {2025},
  eprint        = {2505.12116},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CL},
  url           = {https://arxiv.org/abs/2505.12116}
}
```

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
