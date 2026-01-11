"""
Load and merge data from datasets EmotionRecognition and GoEmotions.
"""

import pandas as pd
import numpy as np
from typing import Dict

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
    try:
        df = pd.read_csv(path)
        if df['label'].dtype in [int, np.int64]:
            df['label'] = df['label'].map(EMOTION_RECOGNITION_LABEL_MAP)
        return df[['text', 'label']]
    except FileNotFoundError:
        print(f"Warning: File not found at {path}")
        return pd.DataFrame(columns=['text', 'label'])


def load_goemotions(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep='\t', header=None, names=['text', 'label_ids', 'id'])
    except FileNotFoundError:
        print(f"Warning: File not found at {path}")
        return pd.DataFrame(columns=['text', 'label'])

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

        for target in found_targets:
            texts.append(row['text'])
            labels.append(target)

    return pd.DataFrame({'text': texts, 'label': labels})


def load_all_data(base_path: str = '', random_state: int = 42) -> Dict[str, pd.DataFrame]:
    paths = {
        'orig_train': f'{base_path}/EmotionRecognitionDataset/training.csv',
        'orig_val':   f'{base_path}/EmotionRecognitionDataset/validation.csv',
        'orig_test':  f'{base_path}/EmotionRecognitionDataset/test.csv',
        'go_train':   f'{base_path}/GoEmotions/data/train.tsv',
        'go_val':     f'{base_path}/GoEmotions/data/dev.tsv',
        'go_test':    f'{base_path}/GoEmotions/data/test.tsv'
    }

    orig_train = load_emotion_recognition(paths['orig_train'])
    orig_val = load_emotion_recognition(paths['orig_val'])
    orig_test = load_emotion_recognition(paths['orig_test'])

    go_train = load_goemotions(paths['go_train'])
    go_val = load_goemotions(paths['go_val'])
    go_test = load_goemotions(paths['go_test'])

    # Combine
    full_train_df = pd.concat([orig_train, orig_val, go_train, go_val], ignore_index=True)
    full_train_df = full_train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    full_train_df = full_train_df.drop_duplicates(subset=['text'])

    full_test_df = pd.concat([orig_test, go_test], ignore_index=True)

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

