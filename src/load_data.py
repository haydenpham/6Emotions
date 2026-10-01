"""
Load and merge data from datasets EmotionRecognition and GoEmotions.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

TARGET_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

EMOTION_RECOGNITION_LABEL_MAP = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

IDX_TO_EMOTION = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement", 14: "fear",
    15: "gratitude", 16: "grief", 17: "joy", 18: "love", 19: "nervousness",
    20: "optimism", 21: "pride", 22: "realization", 23: "relief", 24: "remorse",
    25: "sadness", 26: "surprise", 27: "neutral"
}

GO_MAPPING = {
    "anger": ["anger", "annoyance", "disapproval"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "optimism", "relief", "pride", "admiration", "desire"],
    "love": ["love", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"]
}

REVERSE_GO_MAPPING = {}
for target, sources in GO_MAPPING.items():
    for source in sources:
        REVERSE_GO_MAPPING[source] = target


def load_emotion_recognition(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df['label'].dtype in [int, np.int64]:
        df['label'] = df['label'].map(EMOTION_RECOGNITION_LABEL_MAP)
    return df[['text', 'label']]


def load_goemotions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=None, names=['text', 'label_ids', 'id'])

    texts = []
    labels = []

    for _, row in df.iterrows():
        try:
            id_list = [int(x) for x in str(row['label_ids']).split(',')]
        except ValueError:
            continue

        found_targets = set()
        for emotion_id in id_list:
            if emotion_id not in IDX_TO_EMOTION:
                continue
            raw_emotion = IDX_TO_EMOTION[emotion_id]
            if raw_emotion in REVERSE_GO_MAPPING:
                found_targets.add(REVERSE_GO_MAPPING[raw_emotion])

        for target in TARGET_LABELS:
            if target not in found_targets:
                continue
            texts.append(row['text'])
            labels.append(target)

    return pd.DataFrame({'text': texts, 'label': labels})


def _single_label_texts(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per exact text only when all its labels agree."""
    unambiguous = frame.groupby('text')['label'].transform('nunique').eq(1)
    return frame.loc[unambiguous].drop_duplicates(subset=['text']).reset_index(drop=True)


def load_all_data(base_path: str | Path = 'data', random_state: int = 42) -> Dict[str, pd.DataFrame]:
    paths = {
        'orig_train': f'{base_path}/EmotionRecognitionDataset/training.csv',
        'orig_val':   f'{base_path}/EmotionRecognitionDataset/validation.csv',
        'orig_test':  f'{base_path}/EmotionRecognitionDataset/test.csv',
        'go_train':   f'{base_path}/GoEmotions/data/train.tsv',
        'go_val':     f'{base_path}/GoEmotions/data/dev.tsv',
        'go_test':    f'{base_path}/GoEmotions/data/test.tsv'
    }
    missing = [path for path in paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset files missing: {', '.join(missing)}")

    orig_train = load_emotion_recognition(paths['orig_train'])
    orig_val = load_emotion_recognition(paths['orig_val'])
    orig_test = load_emotion_recognition(paths['orig_test'])

    go_train = load_goemotions(paths['go_train'])
    go_val = load_goemotions(paths['go_val'])
    go_test = load_goemotions(paths['go_test'])

    full_test_df = _single_label_texts(pd.concat([orig_test, go_test], ignore_index=True))

    full_train_df = _single_label_texts(
        pd.concat([orig_train, orig_val, go_train, go_val], ignore_index=True)
    )
    full_train_df = full_train_df.loc[
        ~full_train_df['text'].isin(full_test_df['text'])
    ]
    full_train_df = full_train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return {
        'orig_train': orig_train,
        'orig_val': orig_val,
        'orig_test': orig_test,
        'go_train': go_train,
        'go_val': go_val,
        'go_test': go_test,
        'full_train': full_train_df,
        'full_test': full_test_df
    }


def load_model_data(base_path: str | Path = 'data', random_state: int = 42, joy_cap: int = 12000) -> Dict[str, pd.DataFrame]:
    # Return the shared six-emotion train and test sets for models
    data = load_all_data(base_path=base_path, random_state=random_state)
    full_train = data['full_train']
    joy = full_train[full_train['label'] == 'joy'].sample(n=joy_cap, random_state=random_state)
    other = full_train[full_train['label'] != 'joy']
    full_train = pd.concat([joy, other], ignore_index=True)
    full_train = full_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    full_test = data['full_test']

    # Guard against test leakage
    assert not set(full_train['text']) & set(full_test['text']), 'Test texts found in training data'
    assert not full_test['text'].duplicated().any(), 'Duplicate texts in test data'
    return {'full_train': full_train, 'full_test': full_test}
