import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# читаєм вхідний датасет авто
csv_path = os.path.join(os.path.dirname(__file__), "task", "Automobile_data.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), "Automobile_data.csv")

df = pd.read_csv(csv_path).replace("?", np.nan)

num_cols = [
    "wheel-base", "length", "width", "height", "curb-weight",
    "engine-size", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price", "normalized-losses"
]

cat_cols = [
    "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
    "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system"
]

# приводимо числові ознаки до float та заповнюємо пропуски
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# набір a: всі підготовлені ознаки
prep_a = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])
X_a = prep_a.fit_transform(df[num_cols + cat_cols])

# набір b: відібрані ключові неколінеарні ознаки
sel_num = ["engine-size", "horsepower", "city-mpg", "price", "curb-weight"]
sel_cat = ["body-style", "drive-wheels", "fuel-type"]
prep_b = ColumnTransformer(transformers=[
    ("num", StandardScaler(), sel_num),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), sel_cat)
])
X_b = prep_b.fit_transform(df[sel_num + sel_cat])

# набір c: головні компоненти pca з поясненням 90% дисперсії
pca = PCA(n_components=0.90, random_state=42)
X_c = pca.fit_transform(X_a)

print("features in Set A:", X_a.shape[1])
print("features in Set B:", X_b.shape[1])
print("features in Set C (PCA):", X_c.shape[1])

# досліджуємо оптимальну кількість кластерів k від 2 до 10
k_range = list(range(2, 11))
inertias = []
sil_scores_b = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_b)
    inertias.append(km.inertia_)
    sil_scores_b.append(silhouette_score(X_b, km.labels_))

optimal_k = 3
print("\noptimal k selected:", optimal_k)

# навчаємо kmeans для трьох моделей
km_a = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(X_a)
km_b = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(X_b)
km_c = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(X_c)

df["cluster"] = km_b.labels_

eval_results = [
    {
        "Model": "Model A (All Features)",
        "Features": X_a.shape[1],
        "Inertia": round(km_a.inertia_, 2),
        "Silhouette": round(silhouette_score(X_a, km_a.labels_), 4)
    },
    {
        "Model": "Model B (Selected Features)",
        "Features": X_b.shape[1],
        "Inertia": round(km_b.inertia_, 2),
        "Silhouette": round(silhouette_score(X_b, km_b.labels_), 4)
    },
    {
        "Model": "Model C (PCA 90%)",
        "Features": X_c.shape[1],
        "Inertia": round(km_c.inertia_, 2),
        "Silhouette": round(silhouette_score(X_c, km_c.labels_), 4)
    }
]
eval_df = pd.DataFrame(eval_results)
print("\nclustering quality evaluation:\n", eval_df)

# профілювання кластерів
cluster_profiles = df.groupby("cluster")[["price", "horsepower", "city-mpg", "engine-size", "curb-weight"]].mean().round(1)
cluster_profiles["count"] = df["cluster"].value_counts()
print("\ncluster profiles (means):\n", cluster_profiles)

# графіки
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. elbow method
axes[0, 0].plot(k_range, inertias, marker="o", color="#1f77b4")
axes[0, 0].axvline(optimal_k, color="red", linestyle="--", label=f"optimal k={optimal_k}")
axes[0, 0].set_title("elbow method (inertia vs k)")
axes[0, 0].set_xlabel("number of clusters (k)")
axes[0, 0].set_ylabel("inertia")
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle="--", alpha=0.5)

# 2. silhouette scores
axes[0, 1].plot(k_range, sil_scores_b, marker="s", color="#2ca02c")
axes[0, 1].axvline(optimal_k, color="red", linestyle="--", label=f"optimal k={optimal_k}")
axes[0, 1].set_title("silhouette score vs k")
axes[0, 1].set_xlabel("number of clusters (k)")
axes[0, 1].set_ylabel("silhouette score")
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle="--", alpha=0.5)

# 3. візуалізація кластерів у 2d просторі pca (pc1 x pc2)
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_b)
scatter = axes[1, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=km_b.labels_, cmap="viridis", alpha=0.8, edgecolors="k", s=50)
# центроїди
centers_2d = pca_2d.transform(km_b.cluster_centers_)
axes[1, 0].scatter(centers_2d[:, 0], centers_2d[:, 1], c="red", marker="X", s=200, label="centroids")
axes[1, 0].set_title("clusters in 2d pca space (pc1 x pc2)")
axes[1, 0].set_xlabel(f"pc1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
axes[1, 0].set_ylabel(f"pc2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle="--", alpha=0.5)

# 4. розподіл цін між кластерами
cluster_names = {0: "Budget", 1: "Luxury/Sport", 2: "Mid-Range"}
df["cluster_name"] = df["cluster"].map(cluster_names)
df.boxplot(column="price", by="cluster_name", ax=axes[1, 1], grid=True)
axes[1, 1].set_title("price distribution across clusters")
axes[1, 1].set_xlabel("cluster")
axes[1, 1].set_ylabel("price ($)")
fig.suptitle("")

plt.tight_layout()
plt.show()
