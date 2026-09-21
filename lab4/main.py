import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# зчитуєм вхідний датасет авто
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

# приводимо числові колонки до типу float
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# заповнюємо пропуски медіанами та модами
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# формуємо матрицю ознак X та цільову змінну y (страховий ризик symboling)
X = df[num_cols + cat_cols]
# бінаризуємо: 1 - підвищений ризик (symboling > 0), 0 - безпечний або нейтральний
y = (df["symboling"] > 0).astype(int)

# стратифікований поділ на train та test 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# налаштовуємо пайплайн обробки ознак
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# навчаємо препроцесор тільки на train
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

print("initial features count:", X.shape[1])
print("preprocessed features count:", X_train_prep.shape[1])
print("class distribution in train:\n", y_train.value_counts())

# модель а: всі ознаки
clf_a = RandomForestClassifier(n_estimators=100, random_state=42)
clf_a.fit(X_train_prep, y_train)
y_pred_a = clf_a.predict(X_test_prep)
y_prob_a = clf_a.predict_proba(X_test_prep)[:, 1]

# модель b: відібрані ознаки за feature importance
importances = clf_a.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]
X_train_b = X_train_prep[:, top_idx]
X_test_b = X_test_prep[:, top_idx]

clf_b = RandomForestClassifier(n_estimators=100, random_state=42)
clf_b.fit(X_train_b, y_train)
y_pred_b = clf_b.predict(X_test_b)
y_prob_b = clf_b.predict_proba(X_test_b)[:, 1]

# модель c: pca з поясненням 90% дисперсії
pca = PCA(n_components=0.90, random_state=42)
X_train_c = pca.fit_transform(X_train_prep)
X_test_c = pca.transform(X_test_prep)

clf_c = RandomForestClassifier(n_estimators=100, random_state=42)
clf_c.fit(X_train_c, y_train)
y_pred_c = clf_c.predict(X_test_c)
y_prob_c = clf_c.predict_proba(X_test_c)[:, 1]

# формуємо таблицю оцінки моделей
results = [
    {
        "Model": "Model A (All Features)",
        "Features": X_train_prep.shape[1],
        "Accuracy": round(accuracy_score(y_test, y_pred_a), 4),
        "Precision": round(precision_score(y_test, y_pred_a), 4),
        "Recall": round(recall_score(y_test, y_pred_a), 4),
        "F1": round(f1_score(y_test, y_pred_a), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob_a), 4)
    },
    {
        "Model": "Model B (Selected 15)",
        "Features": 15,
        "Accuracy": round(accuracy_score(y_test, y_pred_b), 4),
        "Precision": round(precision_score(y_test, y_pred_b), 4),
        "Recall": round(recall_score(y_test, y_pred_b), 4),
        "F1": round(f1_score(y_test, y_pred_b), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob_b), 4)
    },
    {
        "Model": "Model C (PCA 90%)",
        "Features": X_train_c.shape[1],
        "Accuracy": round(accuracy_score(y_test, y_pred_c), 4),
        "Precision": round(precision_score(y_test, y_pred_c), 4),
        "Recall": round(recall_score(y_test, y_pred_c), 4),
        "F1": round(f1_score(y_test, y_pred_c), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob_c), 4)
    }
]

res_df = pd.DataFrame(results)
print("\ncomparative model evaluation:\n", res_df)

# графіки результатів
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. топ 10 важливих ознак
top10_idx = np.argsort(importances)[::-1][:10]
top10_names = [feature_names[i].replace("num__", "").replace("cat__", "") for i in top10_idx]
axes[0, 0].barh(range(10), importances[top10_idx][::-1], color="#1f77b4", edgecolor="black")
axes[0, 0].set_yticks(range(10))
axes[0, 0].set_yticklabels(top10_names[::-1], fontsize=8)
axes[0, 0].set_title("top 10 feature importances")
axes[0, 0].grid(True, linestyle="--", alpha=0.5)

# 2. накопичена дисперсія pca
pca_full = PCA().fit(X_train_prep)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
axes[0, 1].plot(range(1, len(cum_var) + 1), cum_var, marker="o", color="#2ca02c", markersize=3)
axes[0, 1].axhline(0.90, color="red", linestyle="--", label="90% variance threshold")
axes[0, 1].set_xlabel("number of components")
axes[0, 1].set_ylabel("cumulative explained variance")
axes[0, 1].set_title("pca explained variance curve")
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle="--", alpha=0.5)

# 3. матриця невідповідностей model b
cm_b = confusion_matrix(y_test, y_pred_b)
im = axes[1, 0].imshow(cm_b, cmap="Blues", interpolation="nearest")
axes[1, 0].set_title("confusion matrix (model b)")
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(["safe (0)", "risky (1)"])
axes[1, 0].set_yticklabels(["safe (0)", "risky (1)"])
for i in range(2):
    for j in range(2):
        axes[1, 0].text(j, i, str(cm_b[i, j]), ha="center", va="center", color="black" if cm_b[i, j] < 15 else "white")

# 4. roc криві
fpr_a, tpr_a, _ = roc_curve(y_test, y_prob_a)
fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_b)
fpr_c, tpr_c, _ = roc_curve(y_test, y_prob_c)

axes[1, 1].plot(fpr_a, tpr_a, label=f"Model A (AUC = {roc_auc_score(y_test, y_prob_a):.2f})", color="#1f77b4")
axes[1, 1].plot(fpr_b, tpr_b, label=f"Model B (AUC = {roc_auc_score(y_test, y_prob_b):.2f})", color="#2ca02c", linestyle="--")
axes[1, 1].plot(fpr_c, tpr_c, label=f"Model C (AUC = {roc_auc_score(y_test, y_prob_c):.2f})", color="#ff7f0e", linestyle=":")
axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[1, 1].set_title("roc curves comparison")
axes[1, 1].set_xlabel("false positive rate")
axes[1, 1].set_ylabel("true positive rate")
axes[1, 1].legend()
axes[1, 1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
