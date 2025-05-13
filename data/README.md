# Tigrinya Abusive Language Detection (TiALD) Dataset

A stable version of TiALD dataset is made available on 🤗 Hugging Face Hub.  

You can head over to: <https://huggingface.co/datasets/fgaim/tigrinya-abusive-language-detection>

Or pull it from anywhere as follows:

```python
from datasets import load_dataset

dataset = load_dataset("fgaim/tigrinya-abusive-language-detection")
print(dataset["validation"][5])  # Inspect a sample
```

## Croissant Metadata for TiALD Dataset

The Croissant of TiALD dataset is at [TiALD.Croissant.json](./TiALD.Croissant.json).
