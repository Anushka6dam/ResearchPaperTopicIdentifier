import pandas as pd
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = "dataset/processed/research_papers.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

# ==========================================
# 2. BASIC INFORMATION
# ==========================================

print("\n===================================")
print("BASIC DATASET INFORMATION")
print("===================================")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

# ==========================================
# 3. FIRST 5 RECORDS
# ==========================================

print("\n===================================")
print("FIRST 5 RECORDS")
print("===================================")

print(df.head())

# ==========================================
# 4. MISSING VALUES
# ==========================================

print("\n===================================")
print("MISSING VALUES")
print("===================================")

print(df.isnull().sum())

# ==========================================
# 5. DUPLICATE RECORDS
# ==========================================

print("\n===================================")
print("DUPLICATE RECORDS")
print("===================================")

duplicates = df.duplicated(subset=["id"]).sum()

print("Duplicate paper IDs:", duplicates)

# ==========================================
# 6. TOPIC DISTRIBUTION
# ==========================================

print("\n===================================")
print("TOPIC DISTRIBUTION")
print("===================================")

topic_counts = df["topic"].value_counts()

print(topic_counts)

# ==========================================
# 7. TITLE LENGTH
# ==========================================

df["title_length"] = df["title"].astype(str).str.len()

print("\n===================================")
print("TITLE LENGTH")
print("===================================")

print("Average title length:",
      round(df["title_length"].mean(), 2))

print("Maximum title length:",
      df["title_length"].max())

print("Minimum title length:",
      df["title_length"].min())

# ==========================================
# 8. ABSTRACT LENGTH
# ==========================================

df["abstract_length"] = df["abstract"].astype(str).str.len()

print("\n===================================")
print("ABSTRACT LENGTH")
print("===================================")

print("Average abstract length:",
      round(df["abstract_length"].mean(), 2))

print("Maximum abstract length:",
      df["abstract_length"].max())

print("Minimum abstract length:",
      df["abstract_length"].min())

# ==========================================
# 9. CREATE RESULTS FOLDER
# ==========================================

os.makedirs("results", exist_ok=True)

# ==========================================
# 10. TOPIC DISTRIBUTION GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

topic_counts.plot(kind="bar")

plt.title("Research Paper Topic Distribution")
plt.xlabel("Research Topic")
plt.ylabel("Number of Papers")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

graph_path = "results/topic_distribution.png"

plt.savefig(graph_path, dpi=300)

plt.close()

print("\nTopic distribution graph saved at:")

print(graph_path)

# ==========================================
# 11. ABSTRACT LENGTH GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

df["abstract_length"].plot(
    kind="hist",
    bins=50
)

plt.title("Abstract Length Distribution")
plt.xlabel("Abstract Length (Characters)")
plt.ylabel("Number of Papers")

plt.tight_layout()

graph_path = "results/abstract_length_distribution.png"

plt.savefig(graph_path, dpi=300)

plt.close()

print("\nAbstract length graph saved at:")

print(graph_path)

# ==========================================
# 12. FINAL SUMMARY
# ==========================================

print("\n===================================")
print("DATASET VERIFICATION COMPLETED")
print("===================================")

print("Total Papers:", len(df))
print("Total Topics:", df["topic"].nunique())
print("Duplicate IDs:", duplicates)

print("\nDataset is ready for preprocessing.")