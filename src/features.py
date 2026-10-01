"""
Shared TF-IDF features for the LogReg and MLP models.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def make_tfidf_features() -> FeatureUnion:
    word_vectorizer = TfidfVectorizer(
        stop_words=None,
        sublinear_tf=True,
        min_df=5,
        max_features=20000,
        ngram_range=(1, 3),
        token_pattern=r'(?u)\b\w\w+\b|!|\?|<3|:\)|:\(' # exclamation indeed matters
    )
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        min_df=5,
        max_features=15000,
    )
    return FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer),
    ])
