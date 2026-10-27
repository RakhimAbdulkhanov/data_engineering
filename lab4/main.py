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

# візуалізація трьох матриць помилок
cm_a = confusion_matrix(y_test, y_pred_a)
cm_b = confusion_matrix(y_test, y_pred_b)
cm_c = confusion_matrix(y_test, y_pred_c)
fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4.2))
cms = [("Model A (All Features)", cm_a), ("Model B (Selected 15)", cm_b), ("Model C (PCA 13)", cm_c)]
for ax, (title, cm) in zip(axes1, cms):
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["безпечний (0)", "ризиковий (1)"])
    ax.set_yticklabels(["безпечний (0)", "ризиковий (1)"])
    ax.set_xlabel("прогноз моделі")
    ax.set_ylabel("фактичний клас")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > 10 else "black", fontsize=12, fontweight="bold")
fig1.tight_layout()

# комплексні графіки: ознаки, pca та roc-криві
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5))

# топ 10 ознак
top10_idx = np.argsort(importances)[::-1][:10]
top10_names = [feature_names[i].replace("num__", "").replace("cat__", "") for i in top10_idx]
axes2[0].barh(range(10), importances[top10_idx][::-1], color="#1f77b4", edgecolor="black")
axes2[0].set_yticks(range(10))
axes2[0].set_yticklabels(top10_names[::-1], fontsize=9)
axes2[0].set_title("ТОП-10 важливих ознак", fontsize=11, fontweight="bold")
axes2[0].set_xlabel("важливість ознаки")
axes2[0].grid(True, linestyle="--", alpha=0.5)

# накопичена дисперсія pca
pca_full = PCA().fit(X_train_prep)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
axes2[1].plot(range(1, len(cum_var) + 1), cum_var, marker="o", color="#2ca02c", markersize=3)
axes2[1].axhline(0.90, color="red", linestyle="--", label="поріг 90% дисперсії")
axes2[1].axvline(13, color="black", linestyle=":", label="13 компонент")
axes2[1].set_xlabel("кількість головних компонент")
axes2[1].set_ylabel("накопичена дисперсія")
axes2[1].set_title("накопичена дисперсія PCA", fontsize=11, fontweight="bold")
axes2[1].legend(loc="lower right")
axes2[1].grid(True, linestyle="--", alpha=0.5)

# roc-криві
fpr_a, tpr_a, _ = roc_curve(y_test, y_prob_a)
fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_b)
fpr_c, tpr_c, _ = roc_curve(y_test, y_prob_c)
axes2[2].plot(fpr_a, tpr_a, label=f"Model A (AUC = {roc_auc_score(y_test, y_prob_a):.4f})", color="#1f77b4")
axes2[2].plot(fpr_b, tpr_b, label=f"Model B (AUC = {roc_auc_score(y_test, y_prob_b):.4f})", color="#2ca02c", linestyle="--")
axes2[2].plot(fpr_c, tpr_c, label=f"Model C (AUC = {roc_auc_score(y_test, y_prob_c):.4f})", color="#ff7f0e", linestyle=":")
axes2[2].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes2[2].set_title("порівняння ROC-кривих", fontsize=11, fontweight="bold")
axes2[2].set_xlabel("частка хибнопозитивних (FPR)")
axes2[2].set_ylabel("частка істиннопозитивних (TPR)")
axes2[2].legend(loc="lower right")
axes2[2].grid(True, linestyle="--", alpha=0.5)

fig2.tight_layout()
plt.show()
