"""
Hierarchical_cluste_Agglomerative_divisive_M4.py
=================================================
Agglomerative (bottom-up) Hierarchical Clustering on the
Aotizhongxin Air Quality Dataset using PM2.5 and related features.

Output
------
output/Hierarchical_Clustering_M4/

Usage
-----
    python src/Hierarchical_cluste_Agglomerative_divisive_M4.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import warnings

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR      = pathlib.Path(__file__).resolve().parent.parent
INPUT_FILE    = BASE_DIR / "dataset" / "PRSA_Data_Aotizhongxin_raw.csv"
OUTPUT_FOLDER = BASE_DIR / "output" / "Hierarchical_Clustering_M4"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("HIERARCHICAL CLUSTERING — AGGLOMERATIVE")
print("AIR QUALITY DATASET - AOTIZHONGXIN STATION")
print("=" * 75)

data = pd.read_csv(INPUT_FILE)
print(f"\nRows: {data.shape[0]}   Columns: {data.shape[1]}")


# ============================================================
# 3. FEATURE SELECTION
# ============================================================

FEATURE_COLUMNS = ["PM2.5", "PM10", "NO2", "CO", "O3", "TEMP", "DEWP", "WSPM"]

X = data[FEATURE_COLUMNS].copy()

for col in FEATURE_COLUMNS:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].fillna(X[col].median())

X = X.replace([np.inf, -np.inf], np.nan).dropna()

print(f"\nFeatures used: {FEATURE_COLUMNS}")
print(f"Records after cleaning: {len(X):,}")


# ============================================================
# 4. SCALE FEATURES
# ============================================================

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nStandardScaler applied.")


# ============================================================
# 5. DENDROGRAM (sample of 500 for speed)
# ============================================================

print("\nGenerating dendrogram (sample of 500 records) …")

sample_idx = np.random.RandomState(42).choice(len(X_scaled), 500, replace=False)
X_sample   = X_scaled[sample_idx]

linked = linkage(X_sample, method="ward")

plt.figure(figsize=(14, 6))
dendrogram(linked, truncate_mode="level", p=5,
           leaf_font_size=8, color_threshold=0)
plt.title("Dendrogram — Ward Linkage (sample 500)", fontsize=13, fontweight="bold")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "dendrogram.png", dpi=300, bbox_inches="tight")
plt.close()
print("Dendrogram saved.")


# ============================================================
# 6. AGGLOMERATIVE CLUSTERING  (K = 4)
# ============================================================

N_CLUSTERS = 4

print(f"\nFitting AgglomerativeClustering (n_clusters={N_CLUSTERS}, linkage=ward) …")

agg_model = AgglomerativeClustering(
    n_clusters=N_CLUSTERS,
    linkage="ward"
)

labels = agg_model.fit_predict(X_scaled)
print("Clustering completed.")


# ============================================================
# 7. METRICS
# ============================================================

sil  = silhouette_score(X_scaled, labels)
cal  = calinski_harabasz_score(X_scaled, labels)
dav  = davies_bouldin_score(X_scaled, labels)

print("\n" + "=" * 75)
print("CLUSTERING METRICS")
print("=" * 75)
print(f"Silhouette Score        : {sil:.4f}")
print(f"Calinski-Harabasz Score : {cal:.4f}")
print(f"Davies-Bouldin Score    : {dav:.4f}")


# ============================================================
# 8. SAVE CLUSTERED DATA
# ============================================================

result_df = X.copy()
result_df["Cluster"] = labels + 1
result_df.to_csv(OUTPUT_FOLDER / "hierarchical_clustered_data.csv", index=False)

metrics_df = pd.DataFrame({
    "Metric":  ["Silhouette_Score", "Calinski_Harabasz_Score", "Davies_Bouldin_Score"],
    "Value":   [sil, cal, dav]
})
metrics_df.to_csv(OUTPUT_FOLDER / "metrics.csv", index=False)
print("\nClustered data and metrics saved.")


# ============================================================
# 9. PCA 2-D VISUALISATION
# ============================================================

pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=4, alpha=0.5, cmap="tab10")
plt.colorbar(scatter, label="Cluster")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Agglomerative Clustering — PCA Visualisation")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "agglomerative_PCA.png", dpi=300, bbox_inches="tight")
plt.close()
print("PCA plot saved.")


# ============================================================
# 10. CLUSTER COUNTS
# ============================================================

print("\n" + "=" * 75)
print("CLUSTER COUNTS")
print("=" * 75)
print(result_df["Cluster"].value_counts().sort_index())


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print(f"OUTPUTS SAVED TO: {OUTPUT_FOLDER}")
print("=" * 75)
print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
