import os
import pandas as pd
import matplotlib.pyplot as plt

# читаєм датасет
csv_path = os.path.join(os.path.dirname(__file__), "task", "iris1.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), "iris1.csv")

df = pd.read_csv(csv_path)

# виводимо базову інфу про датасет
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("dtypes:\n", df.dtypes)
print("\nhead:\n", df.head(3))
print("\ntail:\n", df.tail(3))
print("\nsample:\n", df.sample(3, random_state=42))

# додаємо нові розрахункові колонки
df["sepal_ratio"] = (df["sepal.length"] / df["sepal.width"]).round(2)
df["petal_area_approx"] = (df["petal.length"] * df["petal.width"]).round(2)

# вибірка через loc та iloc
print("\nloc subset:\n", df.loc[:2, ["sepal.length", "variety"]])
print("\niloc subset:\n", df.iloc[:2, [0, 2]])

# умовна фільтрація
setosa_df = df[df["variety"] == "Setosa"]
print("\nsetosa count:", len(setosa_df))

# статистичні зведення
print("\nvalue_counts variety:\n", df["variety"].value_counts())
print("unique varieties:", df["variety"].unique())
print("mean sepal length:", round(df["sepal.length"].mean(), 2))

# групування та агрегація
grouped = df.groupby("variety").agg({
    "sepal.length": ["mean", "max"],
    "petal.length": ["mean", "max"]
})
print("\ngrouped by variety:\n", grouped)

# сортування за спаданням площі пелюстки
sorted_df = df.sort_values(by="petal_area_approx", ascending=False).head(5)
print("\ntop 5 by petal area:\n", sorted_df[["variety", "petal_area_approx"]])

# крос-табуляція з розбиттям довжини чашолистка на 3 інтервали
df["sepal_cat"] = pd.qcut(df["sepal.length"], q=3, labels=["short", "medium", "long"])
ct = pd.crosstab(df["variety"], df["sepal_cat"])
print("\ncrosstab:\n", ct)

# простий графік для інтерактивного показу
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for species, grp in df.groupby("variety"):
    ax[0].scatter(grp["sepal.length"], grp["petal.length"], label=species, alpha=0.8)
ax[0].set_xlabel("sepal length (cm)")
ax[0].set_ylabel("petal length (cm)")
ax[0].set_title("iris sepal vs petal length")
ax[0].legend()
ax[0].grid(True, linestyle="--", alpha=0.5)

df["petal.width"].hist(ax=ax[1], bins=15, color="steelblue", edgecolor="black")
ax[1].set_xlabel("petal width (cm)")
ax[1].set_ylabel("count")
ax[1].set_title("petal width distribution")
ax[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
