import pandas as pd
import re
import os

# ==========================================
# 1. LOAD DATASET
# ==========================================

INPUT_FILE = "dataset/processed/research_papers.csv"
OUTPUT_FILE = "dataset/processed/cleaned_research_papers.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original papers:", len(df))


# ==========================================
# 2. REMOVE MISSING VALUES
# ==========================================

df = df.dropna(subset=["title", "abstract", "topic"])

print("After removing missing values:", len(df))


# ==========================================
# 3. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates(subset=["id"])

print("After removing duplicates:", len(df))


# ==========================================
# 4. TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+", " ", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# ==========================================
# 5. COMBINE TITLE + ABSTRACT
# ==========================================

print("\nCombining title and abstract...")

df["text"] = (
    df["title"].astype(str)
    + " "
    + df["abstract"].astype(str)
)


# ==========================================
# 6. APPLY TEXT CLEANING
# ==========================================

print("Cleaning text...")

df["clean_text"] = df["text"].apply(clean_text)


# ==========================================
# 7. REMOVE EMPTY TEXT
# ==========================================

df = df[df["clean_text"].str.len() > 20]


# ==========================================
# 8. KEEP REQUIRED COLUMNS
# ==========================================

df = df[
    [
        "id",
        "title",
        "abstract",
        "topic",
        "primary_category",
        "published",
        "clean_text"
    ]
]


# ==========================================
# 9. SAVE CLEAN DATASET
# ==========================================

os.makedirs(
    "dataset/processed",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ==========================================
# 10. FINAL INFORMATION
# ==========================================

print("\n===================================")
print("PREPROCESSING COMPLETED")
print("===================================")

print("Final papers:", len(df))

print("\nTopic distribution:")

print(df["topic"].value_counts())

print("\nSaved at:")

print(OUTPUT_FILE)