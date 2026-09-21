import os
import pandas as pd
import matplotlib.pyplot as plt

# зчитуємо дані для bi конвеєра
csv_path = os.path.join(os.path.dirname(__file__), "task", "data_BI.csv")
df = pd.read_csv(csv_path, low_memory=False)

# перетворюємо дати та чистимо типи
df["order_date"] = pd.to_datetime(df["order_date"])
df["total"] = pd.to_numeric(df["total"], errors="coerce")
df["qty_ordered"] = pd.to_numeric(df["qty_ordered"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["discount_amount"] = pd.to_numeric(df["discount_amount"], errors="coerce").fillna(0)

# припускаємо собівартість як 65% від початкової ціни для розрахунку прибутку
df["cost"] = df["price"] * 0.65 * df["qty_ordered"]
df["profit"] = df["total"] - df["cost"]

print("=== базова інформація ===")
print("shape:", df.shape)
print("columns count:", len(df.columns))

print("\n--- прості питання ---")
# 1. загальна виручка за всіма виконаними замовленнями
completed = df[df["status"] == "complete"]
print("1. total completed revenue:", round(completed["total"].sum(), 2))

# 2. виручка за роками
rev_by_year = df.groupby("year")["total"].sum()
print("2. revenue by year:\n", rev_by_year)

# 3. топ 5 категорій за виручкою
top_categories = df.groupby("category")["total"].sum().sort_values(ascending=False).head(5)
print("3. top 5 categories by revenue:\n", top_categories)

# 4. продажі за регіонами
regional_sales = df.groupby("Region")["total"].sum().sort_values(ascending=False)
print("4. sales by region:\n", regional_sales)

# 5. середній чек та середня знижка
print("5. average order value:", round(df["total"].mean(), 2))
print("   average discount percent:", round(df["Discount_Percent"].mean(), 2))

print("\n--- складні питання та pipeline ---")
# 6. щомісячна динаміка виручки та найприбутковіший місяць
monthly = df.groupby(df["order_date"].dt.to_period("M"))["total"].sum()
best_month = monthly.idxmax()
print("6. best month:", best_month, "with revenue:", round(monthly.max(), 2))

# 7. порівняння робочих та вихідних днів за регіонами
df["is_weekend"] = df["order_date"].dt.dayofweek.isin([5, 6])
day_type_analysis = df.groupby(["Region", "is_weekend"])["total"].agg(["mean", "sum"])
print("7. weekend vs weekday sales by region:\n", day_type_analysis)

# 8. вікові категорії та стать
df["age_group"] = pd.cut(df["age"], bins=[0, 30, 50, 100], labels=["<30", "30-50", ">50"])
demo_sales = df.groupby(["age_group", "Gender"], observed=False)["total"].sum().unstack()
print("8. demographic revenue:\n", demo_sales)

# 9. аналітичний pipeline з розрахунком маржинальності
pipeline_res = (
    df[df["status"].isin(["complete", "received"])]
    .groupby(["Region", "category"], as_index=False)
    .agg(
        total_revenue=("total", "sum"),
        total_profit=("profit", "sum"),
        orders_count=("order_id", "count")
    )
    .assign(margin_pct=lambda x: (x["total_profit"] / x["total_revenue"] * 100).round(2))
    .sort_values(["Region", "total_profit"], ascending=[True, False])
    .groupby("Region")
    .head(3)
)
print("9. top 3 profitable categories per region:\n", pipeline_res)

# 10. зведена таблиця платіжних методів та статусів
pivot_payments = pd.pivot_table(
    df,
    values="total",
    index="payment_method",
    columns="status",
    aggfunc="sum",
    fill_value=0
)
print("10. payment methods vs status pivot (top 5):\n", pivot_payments.iloc[:5, :4])

# побудова графіків
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# динаміка по місяцях
monthly.index = monthly.index.astype(str)
axes[0].plot(monthly.index, monthly.values / 1e6, marker="o", color="#1f77b4")
axes[0].set_title("monthly revenue (m$)")
axes[0].set_xticks(range(len(monthly)))
axes[0].set_xticklabels(monthly.index, rotation=45)
axes[0].grid(True, linestyle="--", alpha=0.5)

# продажі за регіонами
axes[1].bar(regional_sales.index, regional_sales.values / 1e6, color="#2ca02c")
axes[1].set_title("revenue by region (m$)")
axes[1].grid(True, linestyle="--", alpha=0.5)

# топ категорії
axes[2].barh(top_categories.index, top_categories.values / 1e6, color="#ff7f0e")
axes[2].set_title("top 5 categories revenue (m$)")
axes[2].invert_yaxis()
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
