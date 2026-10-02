# MLP emotion classifier

This model classifies English text as sadness, joy, love, anger, fear, or surprise. It uses the same word and character TF-IDF features and shared data loader as LogReg.

Run [train.ipynb](https://github.com/haydenpham/6Emotions/blob/main/models/mlp/train.ipynb) from the repository root or this folder after installing the root `requirements.txt`. The notebook trains one fixed MLP on the full training set and evaluates it once on the held-out test set. It uses one hidden layer of 512 ReLU units, `alpha=0.1`, Adam with learning rate 3e-4, batch size 256, no class weighting, and early stopping on 10% of the training rows (patience 5, at most 20 epochs). Training stopped after 10 epochs.

| Metric | Value |
| --- | ---: |
| Test accuracy | 0.7558 |
| Test macro F1 | 0.7179 |
| Training / test rows | 39,718 / 5,421 |

The MLP pipeline is saved locally to Git-ignored `artifacts/6emotions_model.skops`. The `results/` folder contains the training summary, classification report, and these plots:

**Prediction counts**

![MLP confusion matrix showing prediction counts](results/confusion_matrix.png)

**Row normalized**

![MLP row normalized confusion matrix](results/confusion_matrix_normalized.png)

Rows show true emotions and columns show predictions. The count matrix shows the number of examples in each cell; the normalized matrix shows the share within each true emotion.

## Usage

```python
import skops.io as sio
from huggingface_hub import hf_hub_download

path = hf_hub_download("haydenpham/6emotions", "mlp/6emotions_model.skops")
model = sio.load(path, trusted=["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"])
print(model.predict(["I'm so happy today!"]))  # ['joy']
```

To load a locally trained model instead, pass `models/mlp/artifacts/6emotions_model.skops` as the path.
