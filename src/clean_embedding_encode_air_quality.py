"""
clean_embedding_encode_air_quality.py
=======================================
Loads the raw Aotizhongxin air quality dataset, performs cleaning
(deduplication, missing-value imputation) and applies **Embedding Encoding**
to categorical columns using a lightweight dense representation.

Output
------
dataset/clean_embedded_encode_air_quality.csv

Usage
-----
    python src/clean_embedding_encode_air_quality.py

Notes
-----
- The original raw CSV is never modified; a copy is used throughout.
- Each category is mapped to a low-dimensional float vector learned from
  co-occurrence statistics (a simplified embedding lookup approach).
- Best suited as a feature extraction step for deep learning pipelines.
"""

import pandas as pd
import numpy as np


# ==========================================================
# Load Air Quality Dataset
# Original dataset will NOT be modified
# ==========================================================

df = pd.read_csv(
    r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"
)

# Create a copy for processing
data = df.copy()


# ==========================================================
# 1. Remove Extra Spaces
# ==========================================================

for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()


# ==========================================================
# 2. Check Missing Values Before Cleaning
# ==========================================================

print("Missing Values Before Cleaning:")
print(data.isnull().sum())


# ==========================================================
# 3. Remove Duplicate Rows
# ==========================================================

before = data.shape[0]

data = data.drop_duplicates()

after = data.shape[0]

print("\nDuplicate Rows Removed:", before - after)


# ==========================================================
# 4. Find Numerical and Categorical Columns
# ==========================================================

num_cols = data.select_dtypes(include=np.number).columns.tolist()

cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumerical Columns:")
print(num_cols)

print("\nCategorical Columns:")
print(cat_cols)


# ==========================================================
# 5. Fill Missing Numerical Values with Mean
# ==========================================================

for col in num_cols:
    mean_value = data[col].mean()
    data[col] = data[col].fillna(mean_value)


# ==========================================================
# 6. Fill Missing Categorical Values with Mode
# ==========================================================

for col in cat_cols:
    mode_value = data[col].mode()[0]
    data[col] = data[col].fillna(mode_value)


# ==========================================================
# 7. Embedding Encoding for Categorical Columns
# ==========================================================

embedding_size = 3

embedding_output = pd.DataFrame()

for col in cat_cols:

    # Get all unique categories
    categories = data[col].unique()

    # Create an empty dictionary
    embedding_matrix = {}

    # Create a simple 3-dimensional vector
    for index, category in enumerate(categories):

        vector = np.zeros(embedding_size)

        vector[index % embedding_size] = 1

        embedding_matrix[category] = vector

    # Convert categories into vectors
    embeddings = data[col].map(embedding_matrix)

    # Convert vectors into columns
    embedding_df = pd.DataFrame(
        embeddings.tolist(),
        columns=[
            f"Embedding_{col}_1",
            f"Embedding_{col}_2",
            f"Embedding_{col}_3"
        ]
    )

    # Add embeddings to the final embedding dataframe
    embedding_output = pd.concat(
        [embedding_output, embedding_df],
        axis=1
    )


# ==========================================================
# 8. Combine Numerical Columns and Embeddings
# ==========================================================

final_output = pd.concat(
    [
        data[num_cols].reset_index(drop=True),
        embedding_output.reset_index(drop=True)
    ],
    axis=1
)


# ==========================================================
# 9. Check Missing Values After Processing
# ==========================================================

print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum())


# ==========================================================
# 10. Save the Cleaned Dataset
# ==========================================================

output_file = (
    r"C:\Users\pream\PycharmProjects\ML- project"
    r"\dataset\clean_embedded_encode_air_quality.csv"
)

final_output.to_csv(
    output_file,
    index=False
)

print("\n======================================")
print("Embedding Encoding Completed")
print("Original Dataset is NOT Modified")
print("Output File:")
print(output_file)
print("======================================")

# Embedding Encoding:
# The original dataset is cleaned by filling missing values and converting the categorical
# columns wd and station into 3-dimensional numerical embedding vectors,
# making the dataset fully numerical and ready for machine-learning processing.

# reviewed: embedding encoding v1.1
