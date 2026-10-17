from datasets import load_dataset
import pandas as pd
import os

# ==========================================
# SETTINGS
# ==========================================

OUTPUT_FILE = "dataset/processed/research_papers.csv"

TARGET_PER_TOPIC = {
    "cs.LG": 2000,   # Machine Learning
    "cs.CL": 2000,   # NLP
    "cs.CV": 2000,   # Computer Vision
    "cs.AI": 2000,   # Artificial Intelligence
    "cs.CR": 2000,   # Cybersecurity
    "cs.SE": 1500,   # Software Engineering
    "cs.DB": 1500,   # Databases
    "cs.NI": 2000    # Computer Networks
}

TOPIC_NAMES = {
    "cs.LG": "Machine Learning",
    "cs.CL": "Natural Language Processing",
    "cs.CV": "Computer Vision",
    "cs.AI": "Artificial Intelligence",
    "cs.CR": "Cybersecurity",
    "cs.SE": "Software Engineering",
    "cs.DB": "Databases",
    "cs.NI": "Computer Networks"
}

# ==========================================
# CREATE OUTPUT DIRECTORY
# ==========================================

os.makedirs(
    "dataset/processed",
    exist_ok=True
)

# ==========================================
# LOAD DATASET IN STREAMING MODE
# ==========================================

print("Loading Hugging Face dataset...")

dataset = load_dataset(
    "TheDima/arxiv-cs-2022-2026",
    split="train",
    streaming=True
)

# ==========================================
# COLLECT PAPERS
# ==========================================

collected = []

counts = {
    category: 0
    for category in TARGET_PER_TOPIC
}

print("\nCollecting papers...\n")

for paper in dataset:

    category = paper.get("primary_category", "")

    if category not in TARGET_PER_TOPIC:
        continue

    if counts[category] >= TARGET_PER_TOPIC[category]:
        continue

    title = paper.get("title", "")
    abstract = paper.get("summary", "")

    if not title or not abstract:
        continue

    collected.append({
        "id": paper.get("id", ""),
        "title": title.strip(),
        "abstract": abstract.strip(),
        "topic": TOPIC_NAMES[category],
        "primary_category": category,
        "published": paper.get("published", "")
    })

    counts[category] += 1

    print(
        f"{category}: "
        f"{counts[category]}/{TARGET_PER_TOPIC[category]}"
    )

    # Stop when all categories are complete
    if all(
        counts[c] >= TARGET_PER_TOPIC[c]
        for c in TARGET_PER_TOPIC
    ):
        break

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(collected)

# Remove duplicate papers
df = df.drop_duplicates(subset=["id"])

# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ==========================================
# SAVE DATASET
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("DATASET CREATED SUCCESSFULLY")
print("===================================")

print(f"Total papers: {len(df)}")

print("\nTopic distribution:")
print(df["topic"].value_counts())

print(f"\nSaved at:")
print(OUTPUT_FILE)