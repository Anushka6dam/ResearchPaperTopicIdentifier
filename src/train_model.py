import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

INPUT_FILE = "dataset/processed/cleaned_research_papers.csv"

print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total papers:", len(df))


# ==========================================
# 2. REMOVE ANY EMPTY TEXT
# ==========================================

df = df.dropna(subset=["clean_text", "topic"])

X = df["clean_text"]
y = df["topic"]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

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
# 4. TF-IDF FEATURE EXTRACTION
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:",
      X_train_tfidf.shape)

print("TF-IDF testing shape:",
      X_test_tfidf.shape)


# ==========================================
# 5. TRAIN LINEAR SVM
# ==========================================

print("\nTraining Linear SVM...")

model = LinearSVC(
    C=1.0
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training completed!")


# ==========================================
# 6. PREDICTIONS
# ==========================================

print("\nMaking predictions...")

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 7. ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Topic")

plt.ylabel("Actual Topic")

plt.xticks(
    range(len(model.classes_)),
    model.classes_,
    rotation=90
)

plt.yticks(
    range(len(model.classes_)),
    model.classes_
)

plt.colorbar()

plt.tight_layout()

os.makedirs(
    "results",
    exist_ok=True
)

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 10. SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/svm_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


# ==========================================
# 11. SAVE TEST RESULTS
# ==========================================

results = pd.DataFrame({
    "actual_topic": y_test.values,
    "predicted_topic": y_pred
})

results.to_csv(
    "results/test_predictions.csv",
    index=False
)


# ==========================================
# 12. FINAL OUTPUT
# ==========================================

print("\n===================================")
print("TRAINING COMPLETED")
print("===================================")

print("Model saved:")
print("models/svm_model.pkl")

print("\nVectorizer saved:")
print("models/tfidf_vectorizer.pkl")

print("\nConfusion matrix saved:")
print("results/confusion_matrix.png")

print("\nTest predictions saved:")
print("results/test_predictions.csv")