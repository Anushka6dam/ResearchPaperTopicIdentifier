
from flask import Flask, render_template, request
import joblib
import re
from pathlib import Path

app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# Load the saved model and vectorizer once
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def clean_text(text):
    """Clean text using the same rules as the training dataset."""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    title = ""
    abstract = ""

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        abstract = request.form.get("abstract", "").strip()

        if not title or not abstract:
            error = "Please enter both the paper title and abstract."
        else:
            combined_text = clean_text(title + " " + abstract)

            if len(combined_text) <= 20:
                error = "Please enter a longer title and abstract."
            else:
                features = vectorizer.transform([combined_text])
                prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        title=title,
        abstract=abstract
    )


if __name__ == "__main__":
    app.run(debug=True)