# 6Emotions

Classify English text into six emotions: anger, fear, joy, love, sadness, and surprise. The project has TF-IDF models using logistic regression and a small neural network, plus a fine-tuned DistilBERT experiment. The Gradio app in `spaces/` serves LogReg.

```text
data/                     shared source datasets
src/load_data.py          shared label mapping and model data preparation
src/features.py           shared TF-IDF features (LogReg and MLP)
src/evaluate.py           shared confusion matrix plots
notebooks/data_discovery.ipynb
models/
  logreg/                 training, results, local artifacts
  mlp/                    training, results, local artifacts
  distilbert/             training, results, local artifacts and runs
spaces/                   Gradio app and Dockerfile for the Hugging Face Space
```

Each model folder contains `train.ipynb`, a README, and tracked `results/`. Trained weights and intermediate runs go in Git-ignored `artifacts/` and `runs/`.

| Model | Configuration | Test accuracy | Training / test rows |
| --- | --- | ---: | ---: |
| [LogReg](models/logreg/README.md) | TF-IDF, C=1.0, balanced | 0.7449 | 39,718 / 5,421 |
| [MLP](models/mlp/README.md) | TF-IDF, one hidden layer of 64 | 0.7508 | 41,718 / 5,421 |
| [DistilBERT](models/distilbert/README.md) | 3 epochs, lr 2e-5, weighted loss | 0.8273 | 39,718 / 5,421 |

All three models are evaluated on the same 5,421-row test set. The MLP trains with a `joy` cap of 14,000 instead of 12,000, so its training set has 2,000 more `joy` texts than the other two.

## Data and training

All three models use the same [data loader](src/load_data.py). It combines the Emotion Recognition Dataset and GoEmotions, maps labels to six emotions, excludes texts with conflicting labels, and keeps one row per remaining text. It then removes test texts from training and caps `joy`. `load_model_data` asserts that no test text appears in training. Each model trains one fixed configuration on the full training set and is evaluated once on the held-out test set. Each notebook saves a count and a row-normalized confusion matrix to its `results/`. The dataset READMEs are in [EmotionRecognitionDataset](data/EmotionRecognitionDataset/README.md) and [GoEmotions](data/GoEmotions/README.md).

Install `requirements.txt`, then open a model's `train.ipynb` with Jupyter started from the repository root or that model's folder. See the model READMEs for training details and outputs.

## Space

`spaces/` is a Docker Space. The app loads `6emotions_model.skops` from its own folder, so copy `models/logreg/artifacts/6emotions_model.skops` into `spaces/` before building. [upload_to_hub.py](src/upload_to_hub.py) publishes the same file to a Hugging Face model repo; set `repo_id` first.
