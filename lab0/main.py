import os
import pandas as pd
import matplotlib.pyplot as plt

# читаєм датасет
csv_path = os.path.join(os.path.dirname(__file__), "task", "iris1.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), "iris1.csv")

df = pd.read_csv(csv_path)

# 1. інспекція структури та базові методи
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("index:", df.index)
print("dtypes:\n", df.dtypes)
print("\ninfo:")
df.info()
print("\nhead:\n", df.head(3))
print("\ntail:\n", df.tail(3))
print("\nsample:\n", df.sample(3, random_state=42))

# 2. створення нової колонки як суми двох стовпців
df["total_length"] = df["sepal.length"] + df["petal.length"]
df["sepal_ratio"] = (df["sepal.length"] / df["sepal.width"]).round(2)

# 3. вибірка, loc, iloc, set_index та drop
print("\nloc subset:\n", df.loc[:2, ["sepal.length", "variety"]])
print("\niloc subset:\n", df.iloc[:2, [0, 2]])

df_indexed = df.set_index("variety")
print("\nset_index preview:\n", df_indexed.head(2))

df_dropped = df.drop(columns=["total_length"])
print("columns after drop:", df_dropped.columns.tolist())

# умовна фільтрація та query
setosa_df = df[df["variety"] == "Setosa"]
print("setosa count:", len(setosa_df))

query_res = df.query("`sepal.length` > 6.0 and variety == 'Virginica'")
print("query count (>6.0 and Virginica):", len(query_res))

# 4. summary functions та maps
print("\nvalue_counts:\n", df["variety"].value_counts())
print("unique:", df["variety"].unique())
print("sum petal length:", round(df["petal.length"].sum(), 2))
print("mean sepal length:", round(df["sepal.length"].mean(), 2))

# map зі словником та лямбда виразом
species_code = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
df["species_id"] = df["variety"].map(species_code)
df["variety_upper"] = df["variety"].map(lambda x: str(x).upper())
print("\nmapped columns preview:\n", df[["variety", "species_id", "variety_upper"]].head(3))

# 5. групування за однією, двома колонками та сортування
sorted_asc = df.sort_values(by="sepal.length", ascending=True).head(3)
sorted_desc = df.sort_values(by="sepal.length", ascending=False).head(3)
print("\nsorted ascending (min sepal):\n", sorted_asc[["sepal.length", "variety"]])
print("\nsorted descending (max sepal):\n", sorted_desc[["sepal.length", "variety"]])

df["sepal_cat"] = pd.qcut(df["sepal.length"], q=3, labels=["short", "medium", "long"])
grouped_1 = df.groupby("variety")["sepal.length"].mean()
grouped_2 = df.groupby(["variety", "sepal_cat"], observed=False)["petal.length"].mean()
print("\ngrouped by 1 column:\n", grouped_1)
print("\ngrouped by 2 columns:\n", grouped_2)

# багатопараметрична агрегація
agg_res = df.groupby("variety").agg({
    "sepal.length": ["mean", "max", "min"],
    "petal.length": ["mean", "std"]
})
print("\nmulti-parameter agg:\n", agg_res)

# 6. демонстрація впливу параметрів (2 приклади)
# приклад 1: sort_values з ascending=True vs False
# приклад 2: groupby з as_index=True vs as_index=False
grp_as_index_true = df.groupby("variety", as_index=True)["petal.length"].mean()
grp_as_index_false = df.groupby("variety", as_index=False)["petal.length"].mean()
print("\ngroupby as_index=True type:", type(grp_as_index_true))
print("groupby as_index=False type:", type(grp_as_index_false))

# 7. обробка дат (демонстрація для вимоги завдання)
dates_sample = pd.Series(["2023-01-15", "2024-06-01", "2025-09-20"])
parsed_dates = pd.to_datetime(dates_sample)
now = pd.Timestamp.now()
months_ago = (now.year - parsed_dates.dt.year) * 12 + (now.month - parsed_dates.dt.month)
print("\ndates months to today demo:\n", pd.DataFrame({"date": dates_sample, "months_to_today": months_ago}))

# 8. складніші запити: крос-табуляція та топ-значення
ct = pd.crosstab(df["variety"], df["sepal_cat"])
print("\ncrosstab:\n", ct)

top3_per_species = df.groupby("variety").apply(lambda g: g.nlargest(2, "petal.length"), include_groups=False)
print("\ntop 2 largest petal per variety:\n", top3_per_species[["sepal.length", "petal.length"]])

# 9. інтерактивна візуалізація
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
