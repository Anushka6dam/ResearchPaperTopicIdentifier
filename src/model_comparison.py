import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

DATA_PATH = "dataset/processed/cleaned_research_papers.csv"

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["clean_text", "topic"])

X = df["clean_text"]
y = df["topic"]


# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training papers:", len(X_train))
print("Testing papers:", len(X_test))


# ==========================================
# 3. TF-IDF FEATURE EXTRACTION
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF feature extraction completed.")


# ==========================================
# 4. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC(
        C=1.0
    )
}


# ==========================================
# 5. TRAIN AND COMPARE MODELS
# ==========================================

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy * 100,
        "F1 Score": f1 * 100
    })

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2), "%")


# ==========================================
# 6. CREATE RESULT TABLE
# ==========================================

results_df = pd.DataFrame(results)

print("\n===================================")
print("MODEL COMPARISON RESULTS")
print("===================================")

print(results_df.to_string(index=False))


# ==========================================
# 7. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


# ==========================================
# 8. FIND BEST MODEL
# ==========================================

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\n===================================")
print("BEST MODEL")
print("===================================")

print(
    "Best Model:",
    best_model["Model"]
)

print(
    "Accuracy:",
    round(best_model["Accuracy"], 2),
    "%"
)


# ==========================================
# 9. CREATE COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(9, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.show()


print("\nModel comparison completed successfully.")