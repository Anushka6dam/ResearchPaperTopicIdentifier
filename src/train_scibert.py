import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "dataset/processed/cleaned_research_papers.csv"

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["clean_text", "topic"])

print("Total papers:", len(df))


# ==========================================
# 2. ENCODE TOPICS
# ==========================================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["topic"])

print("\nTopics:")
for i, topic in enumerate(label_encoder.classes_):
    print(i, "=", topic)


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["clean_text"].tolist(),
    df["label"].tolist(),
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining papers:", len(train_texts))
print("Testing papers:", len(test_texts))


# ==========================================
# 4. LOAD SCIBERT
# ==========================================

MODEL_NAME = "allenai/scibert_scivocab_uncased"

print("\nLoading SciBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ==========================================
# 5. TOKENIZE TEXT
# ==========================================

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=256
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=256
)


# ==========================================
# 6. CREATE DATASET CLASS
# ==========================================

class ResearchPaperDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):

        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index]
        )

        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ResearchPaperDataset(
    train_encodings,
    train_labels
)

test_dataset = ResearchPaperDataset(
    test_encodings,
    test_labels
)


# ==========================================
# 7. LOAD PRETRAINED SCIBERT
# ==========================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_encoder.classes_)
)


# ==========================================
# 8. TRAINING SETTINGS
# ==========================================

training_args = TrainingArguments(

    output_dir="models/scibert_output",

    num_train_epochs=2,

    per_device_train_batch_size=8,
                                         
    per_device_eval_batch_size=8,

    learning_rate=2e-5,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_steps=100,

    load_best_model_at_end=True,

    report_to="none"
)


# ==========================================
# 9. EVALUATION FUNCTION
# ==========================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = predictions.argmax(axis=1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "f1": f1
    }


# ==========================================
# 10. TRAINER
# ==========================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics
)


# ==========================================
# 11. TRAIN SCIBERT
# ==========================================

print("\nStarting SciBERT training...")

trainer.train()


# ==========================================
# 12. EVALUATE MODEL
# ==========================================

print("\nEvaluating SciBERT...")

results = trainer.evaluate()

print("\n===================================")
print("SCIBERT RESULTS")
print("===================================")

print(
    "Accuracy:",
    round(results["eval_accuracy"] * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(results["eval_f1"] * 100, 2),
    "%"
)


# ==========================================
# 13. SAVE MODEL
# ==========================================

trainer.save_model("models/scibert_model")

tokenizer.save_pretrained(
    "models/scibert_model"
)

print("\nSciBERT model saved successfully.")


# ==========================================
# 14. SAVE LABELS
# ==========================================

label_mapping = pd.DataFrame({
    "label": range(len(label_encoder.classes_)),
    "topic": label_encoder.classes_
})

label_mapping.to_csv(
    "models/scibert_labels.csv",
    index=False
)

print("Label mapping saved.")

print("\nSciBERT training completed successfully.")
