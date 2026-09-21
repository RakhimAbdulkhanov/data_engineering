import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# читаєм датасет авто
csv_path = os.path.join(os.path.dirname(__file__), "task", "Automobile_data.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), "Automobile_data.csv")

df = pd.read_csv(csv_path)

# заміняєм знаки питання на nan
df = df.replace("?", np.nan)

# визначаєм числові та категорійні стовпці
num_cols = [
    "wheel-base", "length", "width", "height", "curb-weight",
    "engine-size", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price", "normalized-losses"
]

cat_cols = [
    "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
    "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system"
]

# перетворюєм числові ознаки на float
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["symboling"] = pd.to_numeric(df["symboling"], errors="coerce")

print("shape:", df.shape)
print("missing values count:\n", df[num_cols].isna().sum()[df[num_cols].isna().sum() > 0])

# базові описові статистики для числових колонок
stats_records = []
for c in num_cols:
    s = df[c].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers_cnt = len(s[(s < lower) | (s > upper)])
    shapiro_stat, shapiro_p = stats.shapiro(s) if len(s) >= 3 else (np.nan, np.nan)

    stats_records.append({
        "feature": c,
        "mean": round(s.mean(), 2),
        "median": round(s.median(), 2),
        "std": round(s.std(), 2),
        "iqr": round(iqr, 2),
        "skewness": round(s.skew(), 2),
        "outliers": outliers_cnt,
        "shapiro_p": round(shapiro_p, 4)
    })

stats_df = pd.DataFrame(stats_records)
print("\nsummary numerical statistics:\n", stats_df)

# кардинальність категорійних колонок
cat_card = [{"feature": c, "nunique": df[c].nunique(), "mode": df[c].mode()[0]} for c in cat_cols]
print("\ncategorical cardinality:\n", pd.DataFrame(cat_card))

# кореляції з ціною
corr = df[num_cols].corr()
print("\ntop correlations with price:\n", corr["price"].sort_values(ascending=False))

# візуалізація розвідувального аналізу
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. розподіл цін з kde
sns.histplot(df["price"].dropna(), kde=True, ax=axes[0, 0], color="#1f77b4", edgecolor="black")
axes[0, 0].set_title("distribution of price")
axes[0, 0].grid(True, linestyle="--", alpha=0.5)

# 2. ціна від приводу авто
sns.boxplot(data=df, x="drive-wheels", y="price", ax=axes[0, 1], hue="drive-wheels", legend=False, palette="Set2")
axes[0, 1].set_title("price by drive-wheels")
axes[0, 1].grid(True, linestyle="--", alpha=0.5)

# 3. кореляційна теплокарта ключових числових ознак
key_num = ["price", "engine-size", "curb-weight", "horsepower", "city-mpg", "highway-mpg", "wheel-base"]
sns.heatmap(df[key_num].corr(), annot=True, fmt=".2f", cmap="Blues", ax=axes[1, 0], cbar=False)
axes[1, 0].set_title("correlation heatmap")

# 4. q-q plot для ціни
stats.probplot(df["price"].dropna(), dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("q-q plot for price")
axes[1, 1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
