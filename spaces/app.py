import gradio as gr
import skops.io as sio

trusted_types = [
    "sklearn.pipeline.Pipeline",
    "sklearn.linear_model._logistic.LogisticRegression",
    "sklearn.feature_extraction.text.TfidfVectorizer",
    "numpy.ndarray",
    "numpy.dtype"
]

model = sio.load("6emotions_model.skops", trusted=trusted_types)

def predict(text):
    proba = model.predict_proba([text])[0]
    return {label: float(p) for label, p in zip(model.classes_, proba)}

demo = gr.Interface(fn=predict, inputs="text", outputs="label")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)