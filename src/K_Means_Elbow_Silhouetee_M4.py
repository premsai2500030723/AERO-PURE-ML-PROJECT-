"""
K_Means_Elbow_Silhouette_M4.py
================================
K-Means and K-Means++ Clustering on the Air Quality Dataset.
Uses Elbow Method and Silhouette Score to find the best K.
Target: PM2.5 clustering using pollution and weather features.

Output
------
output/K_Means_K++Means_Elbow_Silhouette_M4_Outputs/

Usage
-----
    python src/K_Means_Elbow_Silhouette_M4.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. INPUT FILE
# ============================================================

INPUT_FILE = r"C:\Users\pream\PycharmProjects\ML- project\dataset\PRSA_Data_Aotizhongxin_raw.csv"


# ============================================================
# 2. OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER = os.path.join(
    r"C:\Users\pream\PycharmProjects\ML- project",
    "output",
    "K_Means_K++Means_Elbow_Silhouette_M4_Outputs"
)


# ============================================================
# 3. CREATE SEPARATE OUTPUT FOLDERS
# ============================================================

KMEANS_FOLDER     = os.path.join(OUTPUT_FOLDER, "KMeans")
KMEANS_PP_FOLDER  = os.path.join(OUTPUT_FOLDER, "KMeansPlusPlus")
ELBOW_FOLDER      = os.path.join(OUTPUT_FOLDER, "Elbow")
SILHOUETTE_FOLDER = os.path.join(OUTPUT_FOLDER, "Silhouette")

for folder in [
    KMEANS_FOLDER,
    KMEANS_PP_FOLDER,
    ELBOW_FOLDER,
    SILHOUETTE_FOLDER
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("=" * 75)
print("K-MEANS AND K-MEANS++ CLUSTERING")
print("AIR QUALITY DATASET - AOTIZHONGXIN STATION")
print("=" * 75)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        "\nDataset not found.\nCheck this path:\n" + INPUT_FILE
    )

data = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])


# ============================================================
# 5. DISPLAY COLUMNS
# ============================================================

print("\nAvailable columns:")
print("-" * 75)

for i, col in enumerate(data.columns, 1):
    print(i, ".", col)


# ============================================================
# 6. FEATURES FOR CLUSTERING
# ============================================================

# Using key pollution and weather features
feature_columns = [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "DEWP",
    "WSPM"
]

print("\n" + "=" * 75)
print("FEATURES SELECTED FOR CLUSTERING")
print("=" * 75)

for col in feature_columns:
    print("✓", col)


# ============================================================
# 7. SELECT FEATURES
# ============================================================

X = data[feature_columns].copy()


# ============================================================
# 8. CONVERT TO NUMERIC
# ============================================================

for col in feature_columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")


# ============================================================
# 9. HANDLE INFINITE VALUES
# ============================================================

X = X.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 10. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before imputation:")
print(X.isnull().sum())

for col in feature_columns:
    X[col] = X[col].fillna(X[col].median())

print("\nMissing values after imputation:")
print(X.isnull().sum())


# ============================================================
# 11. STANDARDISATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature standardisation completed.")


# ============================================================
# 12. K RANGE
# ============================================================

K_VALUES = range(2, 11)


# ============================================================
# 13. K-MEANS - ELBOW AND SILHOUETTE
# ============================================================

print("\n" + "=" * 75)
print("K-MEANS ANALYSIS")
print("=" * 75)

kmeans_inertia    = []
kmeans_silhouette = []

for k in K_VALUES:

    model = KMeans(
        n_clusters=k,
        init="random",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    inertia    = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    kmeans_inertia.append(inertia)
    kmeans_silhouette.append(silhouette)

    print(
        f"K = {k:2d} | "
        f"Inertia = {inertia:.4f} | "
        f"Silhouette = {silhouette:.4f}"
    )


# ============================================================
# 14. K-MEANS++ - ELBOW AND SILHOUETTE
# ============================================================

print("\n" + "=" * 75)
print("K-MEANS++ ANALYSIS")
print("=" * 75)

kmeans_pp_inertia    = []
kmeans_pp_silhouette = []

for k in K_VALUES:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    inertia    = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    kmeans_pp_inertia.append(inertia)
    kmeans_pp_silhouette.append(silhouette)

    print(
        f"K = {k:2d} | "
        f"Inertia = {inertia:.4f} | "
        f"Silhouette = {silhouette:.4f}"
    )


# ============================================================
# 15. SELECT BEST K
# ============================================================

best_k_kmeans    = list(K_VALUES)[np.argmax(kmeans_silhouette)]
best_k_kmeans_pp = list(K_VALUES)[np.argmax(kmeans_pp_silhouette)]

print("\n" + "=" * 75)
print("SELECTED NUMBER OF CLUSTERS")
print("=" * 75)

print("K-Means best K    :", best_k_kmeans)
print("K-Means++ best K  :", best_k_kmeans_pp)


# ============================================================
# 16. ELBOW GRAPH - K-MEANS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(list(K_VALUES), kmeans_inertia, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - K-Means (Air Quality)")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(ELBOW_FOLDER, "elbow_kmeans.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. ELBOW GRAPH - K-MEANS++
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(list(K_VALUES), kmeans_pp_inertia, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - K-Means++ (Air Quality)")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(ELBOW_FOLDER, "elbow_kmeans_plus_plus.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. SILHOUETTE GRAPH - K-MEANS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(list(K_VALUES), kmeans_silhouette, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - K-Means (Air Quality)")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(SILHOUETTE_FOLDER, "silhouette_kmeans.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. SILHOUETTE GRAPH - K-MEANS++
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(list(K_VALUES), kmeans_pp_silhouette, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - K-Means++ (Air Quality)")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(SILHOUETTE_FOLDER, "silhouette_kmeans_plus_plus.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 20. FINAL K-MEANS MODEL
# ============================================================

kmeans_model = KMeans(
    n_clusters=best_k_kmeans,
    init="random",
    n_init=10,
    random_state=42
)

kmeans_labels = kmeans_model.fit_predict(X_scaled)


# ============================================================
# 21. FINAL K-MEANS++ MODEL
# ============================================================

kmeans_pp_model = KMeans(
    n_clusters=best_k_kmeans_pp,
    init="k-means++",
    n_init=10,
    random_state=42
)

kmeans_pp_labels = kmeans_pp_model.fit_predict(X_scaled)


# ============================================================
# 22. SAVE K-MEANS CLUSTERED DATA
# ============================================================

kmeans_result = X.copy()
kmeans_result["KMeans_Cluster"] = kmeans_labels + 1

kmeans_result.to_csv(
    os.path.join(KMEANS_FOLDER, "kmeans_clustered_data.csv"),
    index=False
)


# ============================================================
# 23. SAVE K-MEANS++ CLUSTERED DATA
# ============================================================

kmeans_pp_result = X.copy()
kmeans_pp_result["KMeansPlusPlus_Cluster"] = kmeans_pp_labels + 1

kmeans_pp_result.to_csv(
    os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_clustered_data.csv"),
    index=False
)


# ============================================================
# 24. FINAL METRICS - K-MEANS
# ============================================================

km_silhouette = silhouette_score(X_scaled, kmeans_labels)
km_calinski   = calinski_harabasz_score(X_scaled, kmeans_labels)
km_davies     = davies_bouldin_score(X_scaled, kmeans_labels)


# ============================================================
# 25. FINAL METRICS - K-MEANS++
# ============================================================

pp_silhouette = silhouette_score(X_scaled, kmeans_pp_labels)
pp_calinski   = calinski_harabasz_score(X_scaled, kmeans_pp_labels)
pp_davies     = davies_bouldin_score(X_scaled, kmeans_pp_labels)


# ============================================================
# 26. SAVE K-MEANS METRICS
# ============================================================

kmeans_metrics = pd.DataFrame({
    "Method":                  ["K-Means"],
    "Number_of_Clusters":      [best_k_kmeans],
    "Inertia":                 [kmeans_model.inertia_],
    "Silhouette_Score":        [km_silhouette],
    "Calinski_Harabasz_Score": [km_calinski],
    "Davies_Bouldin_Score":    [km_davies]
})

kmeans_metrics.to_csv(
    os.path.join(KMEANS_FOLDER, "kmeans_metrics.csv"),
    index=False
)


# ============================================================
# 27. SAVE K-MEANS++ METRICS
# ============================================================

kmeans_pp_metrics = pd.DataFrame({
    "Method":                  ["K-Means++"],
    "Number_of_Clusters":      [best_k_kmeans_pp],
    "Inertia":                 [kmeans_pp_model.inertia_],
    "Silhouette_Score":        [pp_silhouette],
    "Calinski_Harabasz_Score": [pp_calinski],
    "Davies_Bouldin_Score":    [pp_davies]
})

kmeans_pp_metrics.to_csv(
    os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_metrics.csv"),
    index=False
)


# ============================================================
# 28. PCA 2-D VISUALISATION
# ============================================================

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# ============================================================
# 29. K-MEANS PCA GRAPH
# ============================================================

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    s=5,
    alpha=0.5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters - PCA Visualization (Air Quality)")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(KMEANS_FOLDER, "kmeans_PCA_clustering.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 30. K-MEANS++ PCA GRAPH
# ============================================================

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_pp_labels,
    s=5,
    alpha=0.5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means++ Clusters - PCA Visualization (Air Quality)")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_PCA_clustering.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 31. 3-D K-MEANS CLUSTERING GRAPH
# ============================================================

fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X[feature_columns[0]],
    X[feature_columns[1]],
    X[feature_columns[2]],
    c=kmeans_labels,
    s=5,
    alpha=0.5
)

ax.set_xlabel(feature_columns[0])
ax.set_ylabel(feature_columns[1])
ax.set_zlabel(feature_columns[2])
ax.set_title("K-Means Clustering (Air Quality)")
fig.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()

plt.savefig(
    os.path.join(KMEANS_FOLDER, "kmeans_3D_clustering.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 32. 3-D K-MEANS++ CLUSTERING GRAPH
# ============================================================

fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X[feature_columns[0]],
    X[feature_columns[1]],
    X[feature_columns[2]],
    c=kmeans_pp_labels,
    s=5,
    alpha=0.5
)

ax.set_xlabel(feature_columns[0])
ax.set_ylabel(feature_columns[1])
ax.set_zlabel(feature_columns[2])
ax.set_title("K-Means++ Clustering (Air Quality)")
fig.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()

plt.savefig(
    os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_3D_clustering.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 33. CLUSTER COUNTS
# ============================================================

print("\n" + "=" * 75)
print("K-MEANS CLUSTER COUNTS")
print("=" * 75)

print(
    kmeans_result["KMeans_Cluster"].value_counts().sort_index()
)

print("\n" + "=" * 75)
print("K-MEANS++ CLUSTER COUNTS")
print("=" * 75)

print(
    kmeans_pp_result["KMeansPlusPlus_Cluster"].value_counts().sort_index()
)


# ============================================================
# 34. FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL CLUSTERING RESULTS")
print("=" * 75)

print("\nK-MEANS")
print("-" * 40)
print("Best K              :", best_k_kmeans)
print("Inertia             :", round(kmeans_model.inertia_, 4))
print("Silhouette          :", round(km_silhouette, 4))
print("Calinski-Harabasz   :", round(km_calinski, 4))
print("Davies-Bouldin      :", round(km_davies, 4))

print("\nK-MEANS++")
print("-" * 40)
print("Best K              :", best_k_kmeans_pp)
print("Inertia             :", round(kmeans_pp_model.inertia_, 4))
print("Silhouette          :", round(pp_silhouette, 4))
print("Calinski-Harabasz   :", round(pp_calinski, 4))
print("Davies-Bouldin      :", round(pp_davies, 4))


# ============================================================
# 35. OUTPUT LOCATION
# ============================================================

print("\n" + "=" * 75)
print("OUTPUTS SAVED")
print("=" * 75)

print("\nMain folder      :", OUTPUT_FOLDER)
print("K-Means          :", KMEANS_FOLDER)
print("K-Means++        :", KMEANS_PP_FOLDER)
print("Elbow            :", ELBOW_FOLDER)
print("Silhouette       :", SILHOUETTE_FOLDER)

print("\nOriginal dataset was NOT modified.")
print("\nPROGRAM COMPLETED SUCCESSFULLY.")
