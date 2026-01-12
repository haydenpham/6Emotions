import gradio as gr
import skops.io as sio
import numpy as np

trusted_types = [
    "sklearn.pipeline.Pipeline",
    "sklearn.linear_model._logistic.LogisticRegression",
    "sklearn.feature_extraction.text.TfidfVectorizer",
    "sklearn.pipeline.FeatureUnion",
    "numpy.ndarray",
    "numpy.dtype"
]

model = sio.load("6emotions_model.skops", trusted=trusted_types)


def predict_with_explanation(text):
    # Standard Prediction
    proba = model.predict_proba([text])[0]
    pred_label_dict = {label: float(p) for label, p in zip(model.classes_, proba)}

    # Get the top predicted label
    top_label = max(pred_label_dict, key=pred_label_dict.get)
    class_idx = list(model.classes_).index(top_label)

    # Explanation Logic

    # 1. Get the components from the loaded pipeline
    feature_union = model.named_steps['features']
    classifier = model.named_steps['logreg']

    # 2. Vectorize the input to find which features are present
    vectorized_input = feature_union.transform([text])

    # 3. Reconstruct feature names (Word + Char)
    # Note: We access the transformers by index based on your training code structure
    word_vectorizer = feature_union.transformer_list[0][1]
    char_vectorizer = feature_union.transformer_list[1][1]

    word_names = word_vectorizer.get_feature_names_out()
    char_names = char_vectorizer.get_feature_names_out()
    all_feature_names = np.r_[word_names, char_names]

    # 4. Calculate contribution scores
    active_indices = vectorized_input.indices
    active_values = vectorized_input.data
    class_coefficients = classifier.coef_[class_idx]

    explanation_data = []
    for idx, tfidf_val in zip(active_indices, active_values):
        score = tfidf_val * class_coefficients[idx]
        # Only care about Positive contributors
        if score > 0:
            explanation_data.append({
                "token": str(all_feature_names[idx]),
                "score": float(score)
            })

    # Sort by highest, take top 5
    explanation_data = sorted(explanation_data, key=lambda x: x['score'], reverse=True)[:5]

    return pred_label_dict, explanation_data

demo = gr.Interface(
    fn=predict_with_explanation,
    inputs="text",
    outputs=[
        gr.Label(label="Emotion"),
        gr.JSON(label="Why? (Top Trigger Words)")
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)