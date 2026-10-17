import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================
# 1. PATHS
# ==========================================

DATA_PATH = "dataset/processed/cleaned_research_papers.csv"

MODEL_PATH = "models/scibert_output/checkpoint-1500"


# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["clean_text", "topic"])

print("Total papers:", len(df))


# ==========================================
# 3. ENCODE TOPICS
# ==========================================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["topic"])

print("\nTopics:")

for i, topic in enumerate(label_encoder.classes_):
    print(i, "=", topic)


# ==========================================
# 4. SAME TRAIN-TEST SPLIT
# ==========================================

_, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

test_texts = test_df["clean_text"].tolist()
test_labels = test_df["label"].tolist()

print("\nTesting papers:", len(test_texts))


# ==========================================
# 5. LOAD TOKENIZER
# ==========================================

print("\nLoading SciBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "allenai/scibert_scivocab_uncased"
)


# ==========================================
# 6. LOAD TRAINED CHECKPOINT
# ==========================================

print("\nLoading checkpoint-1500...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(label_encoder.classes_)
)

model.eval()


# ==========================================
# 7. PREDICTION
# ==========================================

predictions = []

batch_size = 8

print("\nStarting evaluation...")


for i in range(0, len(test_texts), batch_size):

    batch_texts = test_texts[i:i + batch_size]

    encoded = tokenizer(
        batch_texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**encoded)

        batch_predictions = torch.argmax(
            outputs.logits,
            dim=1
        ).cpu().numpy()

    predictions.extend(batch_predictions)

    if (i + batch_size) % 400 == 0:
        print(
            "Processed:",
            min(i + batch_size, len(test_texts)),
            "/",
            len(test_texts)
        )


# ==========================================
# 8. CALCULATE RESULTS
# ==========================================

accuracy = accuracy_score(
    test_labels,
    predictions
)

f1 = f1_score(
    test_labels,
    predictions,
    average="weighted"
)


# ==========================================
# 9. PRINT RESULTS
# ==========================================

print("\n===================================")
print("SCIBERT RESULTS")
print("===================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(f1 * 100, 2),
    "%"
)


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        test_labels,
        predictions,
        target_names=label_encoder.classes_
    )
)


# ==========================================
# 11. SAVE RESULTS
# ==========================================

results = pd.DataFrame({
    "Actual": [
        label_encoder.inverse_transform([x])[0]
        for x in test_labels
    ],

    "Predicted": [
        label_encoder.inverse_transform([x])[0]
        for x in predictions
    ]
})

results.to_csv(
    "results/scibert_predictions.csv",
    index=False
)


print("\nPredictions saved to:")
print("results/scibert_predictions.csv")

print("\nSciBERT evaluation completed successfully.")