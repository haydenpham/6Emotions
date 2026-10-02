---
license: mit
language:
  - en
pipeline_tag: text-classification
tags:
  - text-classification
  - emotion-detection
  - sklearn
  - skops
  - transformers
metrics:
  - accuracy
---

# 6Emotions classifiers

Three models that classify English text as sadness, joy, love, anger, fear, or surprise. Each model is in its own folder with a README, its results, and its weights. The training code is on [GitHub](https://github.com/haydenpham/6Emotions).

| Model | Configuration | Test accuracy |
| --- | --- | ---: |
| [LogReg](logreg/README.md) | Word and character TF-IDF, logistic regression, C=1.0, balanced | 0.7449 |
| [MLP](mlp/README.md) | Same TF-IDF features, one hidden layer of 512 ReLU units | 0.7558 |
| [DistilBERT](distilbert/README.md) | Fine-tuned `distilbert-base-uncased`, 3 epochs, lr 2e-5, weighted loss | 0.8273 |

All three models train on the same 39,718 rows and are evaluated on the same 5,421-row held-out test set. The data combines the Emotion Recognition Dataset and GoEmotions, with labels mapped to six emotions. Texts with conflicting labels are dropped, `joy` is capped at 12,000 training rows, and no test text appears in training.

## Usage

DistilBERT loads with `transformers`:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("haydenpham/6emotions", subfolder="distilbert")
model = AutoModelForSequenceClassification.from_pretrained("haydenpham/6emotions", subfolder="distilbert")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
print(classifier("I'm so happy today!"))  # [{'label': 'joy', ...}]
```

The TF-IDF models are scikit-learn pipelines saved with `skops`:

```python
import skops.io as sio
from huggingface_hub import hf_hub_download

path = hf_hub_download("haydenpham/6emotions", "mlp/6emotions_model.skops")
model = sio.load(path, trusted=["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"])
print(model.predict(["I'm so happy today!"]))  # ['joy']
```

For LogReg, download `logreg/6emotions_model.skops` instead and pass `trusted=[]`.

## Limitations

These are single-label English classifiers. Performance on other languages and on formal or technical writing has not been evaluated. Mapping GoEmotions labels such as `approval` to joy and `curiosity` to surprise adds label noise, which limits accuracy for every model.
