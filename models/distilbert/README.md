# DistilBERT emotion classifier

A fine-tuned `distilbert-base-uncased` sequence classifier for anger, fear, joy, love, sadness, and surprise. DistilBERT is a smaller version of BERT that learned language patterns from large amounts of text. Fine-tuning teaches it to use those patterns to choose an emotion for a sentence.

Run [train.ipynb](https://github.com/haydenpham/6Emotions/blob/main/models/distilbert/train.ipynb) after installing the repository's root `requirements.txt`. Start Jupyter from the repository root or this model folder. The notebook trains once on all 39,718 capped training rows with class-weighted loss and a fixed learning rate of 2e-5, then evaluates on the 5,421-row held-out test set. This can still take a long time on CPU.

| Metric | Value |
| --- | ---: |
| Training rows | 39,718 |
| Test rows | 5,421 |
| Learning rate | 2e-5 |
| Held-out test accuracy | 0.8273 |

The notebook writes the [training summary](results/training_summary.json) and confusion matrices to `results/`. It saves the model and tokenizer to Git-ignored `artifacts/`. Intermediate Trainer files go to `runs/`.

![DistilBERT row normalized confusion matrix](results/confusion_matrix_normalized.png)

Rows show true emotions and columns show the share of each true emotion predicted as each label.

This is an English, single-label classifier. Performance on other languages or writing styles has not been evaluated.

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("haydenpham/6emotions", subfolder="distilbert")
model = AutoModelForSequenceClassification.from_pretrained("haydenpham/6emotions", subfolder="distilbert")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
print(classifier("I'm so happy today!"))  # [{'label': 'joy', ...}]
```

To load a locally trained model instead, pass `"models/distilbert/artifacts"` as the path and drop `subfolder`.
