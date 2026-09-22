import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# читаєм датасет
csv_path = os.path.join(os.path.dirname(__file__), "task", "iris1.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), "iris1.csv")

df = pd.read_csv(csv_path)

# 1. інспекція структури та базові методи
print("=== 1. структура та базові методи ===")
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("index:", df.index)
print("dtypes:\n", df.dtypes)
print("\ninfo:")
df.info()
print("\nhead(3):\n", df.head(3))
print("\ntail(3):\n", df.tail(3))
print("\nsample(3):\n", df.sample(3, random_state=42))

# 2. створення нових колонок
df["total_length"] = df["sepal.length"] + df["petal.length"]
df["sepal_ratio"] = (df["sepal.length"] / df["sepal.width"]).round(2)
print("\nновий стовпець суми total_length:\n", df[["sepal.length", "petal.length", "total_length"]].head(3))

# 3. вибірка, loc, iloc, set_index та drop
print("\n=== 2. вибірка, індексація та фільтрація ===")
print("loc[:2, ['sepal.length', 'variety']]:\n", df.loc[:2, ["sepal.length", "variety"]])
print("\niloc[:2, [0, 2]]:\n", df.iloc[:2, [0, 2]])

df_indexed = df.set_index("variety")
print("\nset_index('variety') preview:\n", df_indexed.head(2))

df_dropped = df.drop(columns=["total_length"])
print("колонки після drop('total_length'):", df_dropped.columns.tolist())

# умовна фільтрація та query
setosa_subset = df[df["variety"] == "Setosa"]
print("кількість записів Setosa через conditional selection:", len(setosa_subset))

query_res = df.query("`sepal.length` > 6.0 and variety == 'Virginica'")
print("кількість записів через query (>6.0 and Virginica):", len(query_res))

# 4. summary functions та maps
print("\n=== 3. підсумкові функції та map ===")
print("value_counts:\n", df["variety"].value_counts())
print("unique:", df["variety"].unique())
print("sum petal length:", round(df["petal.length"].sum(), 2))
print("mean sepal length:", round(df["sepal.length"].mean(), 2))

# map зі словником та лямбда виразом
species_code = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
df["species_id"] = df["variety"].map(species_code)
df["variety_upper"] = df["variety"].map(lambda x: str(x).upper())
print("\nmap зі словником та lambda:\n", df[["variety", "species_id", "variety_upper"]].head(3))

# 5. групування за 1, 2 та 3 колонками, сортування
print("\n=== 4. групування та сортування ===")
sorted_asc = df.sort_values(by="sepal.length", ascending=True).head(3)
sorted_desc = df.sort_values(by="sepal.length", ascending=False).head(3)
print("сортування ascending=True (найменші):\n", sorted_asc[["sepal.length", "variety"]])
print("сортування ascending=False (найбільші):\n", sorted_desc[["sepal.length", "variety"]])

df["sepal_cat"] = pd.qcut(df["sepal.length"], q=3, labels=["short", "medium", "long"])
df["petal_cat"] = pd.qcut(df["petal.length"], q=2, labels=["small", "large"])

# групування за 1 колонкою
print("\nгрупування за 1 колонкою (variety):\n", df.groupby("variety")["sepal.length"].mean().round(2))

# групування за 2 колонками (variety, sepal_cat)
print("\nгрупування за 2 колонками (variety, sepal_cat):\n", df.groupby(["variety", "sepal_cat"], observed=False)["petal.length"].mean().round(2))

# групування за 3 колонками (variety, sepal_cat, petal_cat)
grp_3 = df.groupby(["variety", "sepal_cat", "petal_cat"], observed=False)["sepal.width"].agg(["count", "mean"]).dropna()
print("\nгрупування за 3 колонками (variety, sepal_cat, petal_cat):\n", grp_3.head(6))

# агрегація agg
agg_res = df.groupby("variety").agg({
    "sepal.length": ["mean", "max", "min"],
    "petal.length": ["mean", "std"]
}).round(2)
print("\nбагатопараметрична агрегація agg:\n", agg_res)

# 6. демонстрація впливу параметрів (2 приклади)
print("\n=== 5. вплив параметрів ===")
# приклад 1: sort_values ascending=True vs False
print("приклад 1: sort_values ascending=True vs False змінює порядок сортування від мінімуму або від максимуму")
# приклад 2: groupby as_index=True vs as_index=False
grp_idx_true = df.groupby("variety", as_index=True)["petal.length"].mean()
grp_idx_false = df.groupby("variety", as_index=False)["petal.length"].mean()
print("приклад 2: groupby as_index=True повертає тип:", type(grp_idx_true), "(індекс = категорія)")
print("           groupby as_index=False повертає тип:", type(grp_idx_false), "(звичайний числовий індекс)")

# 7. робота з датами
print("\n=== 6. робота з датами ===")
# створюємо колонку дати збору зразків
np.random.seed(42)
start_date = pd.Timestamp("2024-01-01")
df["collection_date"] = [start_date + pd.Timedelta(days=int(i * 4.5)) for i in range(len(df))]
now = pd.Timestamp.now()
df["months_to_today"] = (now.year - df["collection_date"].dt.year) * 12 + (now.month - df["collection_date"].dt.month)
print("колонки дати та розрахунку місяців до сьогодні:\n", df[["variety", "collection_date", "months_to_today"]].iloc[[0, 75, 149]])

# 8. складніші запити: крос-табуляція та топ-значення
print("\n=== 7. складніші запити ===")
ct = pd.crosstab(df["variety"], df["sepal_cat"])
print("крос-табуляція crosstab:\n", ct)

top2_per_species = df.groupby("variety").apply(lambda g: g.nlargest(2, "petal.length"), include_groups=False)
print("\nтоп-2 найбільші пелюстки для кожного сорту:\n", top2_per_species[["sepal.length", "petal.length"]])

# 9. візуалізація біометричних характеристик
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for species, grp in df.groupby("variety"):
    ax[0].scatter(grp["sepal.length"], grp["petal.length"], label=species, alpha=0.8)
ax[0].set_xlabel("sepal length (cm)")
ax[0].set_ylabel("petal length (cm)")
ax[0].set_title("iris sepal vs petal length")
ax[0].legend()
ax[0].grid(True, linestyle="--", alpha=0.5)

df["petal.width"].hist(ax=ax[1], bins=15, color="#2b5c8f", edgecolor="black")
ax[1].set_xlabel("petal width (cm)")
ax[1].set_ylabel("count")
ax[1].set_title("petal width distribution")
ax[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
