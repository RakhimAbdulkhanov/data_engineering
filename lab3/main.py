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

df = pd.read_csv(csv_path).replace("?", np.nan)

# класифікація колонок за семантикою
numeric_columns = [
    "wheel-base", "length", "width", "height", "curb-weight",
    "engine-size", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price", "normalized-losses"
]

categorical_columns = [
    "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
    "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system"
]

# датасет не містить часових колонок та штучних ідентифікаторів
datetime_columns = []
id_columns = []

# приводимо числові типи до float
for c in numeric_columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["symboling"] = pd.to_numeric(df["symboling"], errors="coerce")

print("=== 1. первинне дослідження (data understanding) ===")
print("shape:", df.shape)
print("memory usage:", df.memory_usage().sum(), "bytes")
print("missing values:\n", df[numeric_columns].isna().sum()[df[numeric_columns].isna().sum() > 0])

print("\n=== 2. повна описова статистика числових ознак ===")
stat_rows = []
for c in numeric_columns:
    s = df[c].dropna()
    q1 = s.quantile(0.25)
    q2 = s.quantile(0.50)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    rng = s.max() - s.min()
    stat_rows.append({
        "feature": c, "count": len(s), "mean": round(s.mean(), 1), "median": round(q2, 1),
        "std": round(s.std(), 1), "min": round(s.min(), 1), "max": round(s.max(), 1),
        "Q1": round(q1, 1), "Q3": round(q3, 1), "range": round(rng, 1),
        "IQR": round(iqr, 1), "skew": round(s.skew(), 2), "kurt": round(s.kurt(), 2)
    })
print(pd.DataFrame(stat_rows))

print("\n=== 3. аналіз категорійних ознак та кардинальність ===")
for c in categorical_columns:
    n_unq = df[c].nunique()
    card_type = "Low" if n_unq <= 3 else ("Medium" if n_unq <= 8 else "High")
    mode_val = df[c].mode()[0]
    rarest_val = df[c].value_counts().idxmin()
    print(f"{c:18} | unique: {n_unq:2} | cardinality: {card_type:6} | mode: {mode_val:10} | rarest: {rarest_val}")

print("\n=== 4. нормальність (тест шапіро-вілка) ===")
shapiro_rows = []
for c in numeric_columns:
    s = df[c].dropna()
    stat_val, p_val = stats.shapiro(s)
    shapiro_rows.append({"feature": c, "skewness": round(s.skew(), 2), "shapiro_p": round(p_val, 5)})
print(pd.DataFrame(shapiro_rows))

print("\n=== 5. аналіз викидів за методом iqr ===")
for c in ["price", "engine-size", "horsepower", "compression-ratio"]:
    s = df[c].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    out_cnt = len(s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)])
    print(f"outliers in {c}: {out_cnt} ({out_cnt/len(s)*100:.1f}%)")

print("\n=== 6. кореляційний аналіз ===")
corr = df[numeric_columns].corr()
print("top 3 positive correlations with price:\n", corr["price"].sort_values(ascending=False).iloc[1:4])
print("top 3 negative correlations with price:\n", corr["price"].sort_values(ascending=True).iloc[:3])

print("\n=== 7. групування числова - категорійна ознака ===")
print("price by drive-wheels:\n", df.groupby("drive-wheels", observed=False)["price"].agg(["count", "mean", "median", "std"]).round(1))
print("\nhorsepower by body-style:\n", df.groupby("body-style", observed=False)["horsepower"].agg(["count", "mean", "median", "std"]).round(1))

# візуалізація розвідувального аналізу
fig = plt.figure(figsize=(14, 9))

# 1. гістограма та kde ціни
ax1 = plt.subplot(2, 3, 1)
sns.histplot(df["price"].dropna(), kde=True, ax=ax1, color="#1f77b4", edgecolor="black")
ax1.set_title("distribution of price")
ax1.grid(True, linestyle="--", alpha=0.5)

# 2. q-q plots для 3 ознак (price, horsepower, curb-weight)
ax2 = plt.subplot(2, 3, 2)
stats.probplot(df["price"].dropna(), dist="norm", plot=ax2)
ax2.set_title("q-q plot: price")
ax2.grid(True, linestyle="--", alpha=0.5)

ax3 = plt.subplot(2, 3, 3)
stats.probplot(df["horsepower"].dropna(), dist="norm", plot=ax3)
ax3.set_title("q-q plot: horsepower")
ax3.grid(True, linestyle="--", alpha=0.5)

ax4 = plt.subplot(2, 3, 4)
stats.probplot(df["curb-weight"].dropna(), dist="norm", plot=ax4)
ax4.set_title("q-q plot: curb-weight")
ax4.grid(True, linestyle="--", alpha=0.5)

# 3. кореляційна матриця
ax5 = plt.subplot(2, 3, 5)
key_num = ["price", "engine-size", "curb-weight", "horsepower", "city-mpg", "highway-mpg"]
sns.heatmap(df[key_num].corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax5, cbar=False, annot_kws={"size": 8})
ax5.set_title("correlation heatmap")

# 4. boxplot числова від категорійної
ax6 = plt.subplot(2, 3, 6)
sns.boxplot(data=df, x="drive-wheels", y="price", ax=ax6, hue="drive-wheels", legend=False, palette="Set2")
ax6.set_title("price by drive-wheels")
ax6.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# матриця розсіювання pairplot
pair_cols = ["price", "engine-size", "horsepower", "city-mpg"]
sns.pairplot(df[pair_cols].dropna(), diag_kind="kde", corner=True)
plt.show()
