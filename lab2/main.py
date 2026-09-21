import os
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# читаєм та готуємо датасет
csv_path = os.path.join(os.path.dirname(__file__), "task", "data_BI.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(os.path.dirname(__file__), "data_BI.csv")

df = pd.read_csv(csv_path, low_memory=False)

# чистимо типи даних
df["order_date"] = pd.to_datetime(df["order_date"])
df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0)
df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
df["qty_ordered"] = pd.to_numeric(df["qty_ordered"], errors="coerce").fillna(0)
df["discount_amount"] = pd.to_numeric(df["discount_amount"], errors="coerce").fillna(0)
df["Discount_Percent"] = pd.to_numeric(df["Discount_Percent"], errors="coerce").fillna(0)

# рахуєм собівартість та чистий прибуток
df["cost"] = df["price"] * 0.65 * df["qty_ordered"]
df["profit"] = df["total"] - df["cost"]
df["month_year"] = df["order_date"].dt.to_period("M").astype(str)

# створюєм dash додаток
app = Dash(__name__)
app.title = "E-Commerce Analytics Dashboard"

regions_list = ["Всі регіони"] + sorted([r for r in df["Region"].dropna().unique()])
categories_list = ["Всі категорії"] + sorted([c for c in df["category"].dropna().unique()])

app.layout = html.Div(
    style={"fontFamily": "Segoe UI, sans-serif", "backgroundColor": "#f4f6f9", "padding": "20px"},
    children=[
        html.Div(
            style={"textAlign": "center", "marginBottom": "20px"},
            children=[
                html.H1("Аналітичний дашборд показників електронної комерції", style={"color": "#2c3e50"}),
                html.P("Інтерактивний моніторинг фінансових показників, продажів та структури замовлень", style={"color": "#7f8c8d"})
            ]
        ),
        # панель фільтрів
        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "20px", "backgroundColor": "#ffffff", "padding": "15px", "borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"},
            children=[
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Label("Регіон:", style={"fontWeight": "bold", "color": "#34495e"}),
                        dcc.Dropdown(id="region-filter", options=[{"label": r, "value": r} for r in regions_list], value="Всі регіони", clearable=False)
                    ]
                ),
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Label("Категорія товару:", style={"fontWeight": "bold", "color": "#34495e"}),
                        dcc.Dropdown(id="category-filter", options=[{"label": c, "value": c} for c in categories_list], value="Всі категорії", clearable=False)
                    ]
                )
            ]
        ),
        # kpi картки
        html.Div(
            id="kpi-cards",
            style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "15px", "marginBottom": "20px"}
        ),
        # графіки перший рядок
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
            children=[
                html.Div(style={"backgroundColor": "#fff", "padding": "10px", "borderRadius": "8px"}, children=[dcc.Graph(id="timeline-chart")]),
                html.Div(style={"backgroundColor": "#fff", "padding": "10px", "borderRadius": "8px"}, children=[dcc.Graph(id="category-chart")])
            ]
        ),
        # графіки другий рядок
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
            children=[
                html.Div(style={"backgroundColor": "#fff", "padding": "10px", "borderRadius": "8px"}, children=[dcc.Graph(id="region-status-chart")]),
                html.Div(style={"backgroundColor": "#fff", "padding": "10px", "borderRadius": "8px"}, children=[dcc.Graph(id="scatter-discount-chart")])
            ]
        )
    ]
)

# колбек оновлення дашборду
@app.callback(
    [
        Output("kpi-cards", "children"),
        Output("timeline-chart", "figure"),
        Output("category-chart", "figure"),
        Output("region-status-chart", "figure"),
        Output("scatter-discount-chart", "figure")
    ],
    [Input("region-filter", "value"), Input("category-filter", "value")]
)
def update_dashboard(selected_region, selected_category):
    filtered = df.copy()
    if selected_region != "Всі регіони":
        filtered = filtered[filtered["Region"] == selected_region]
    if selected_category != "Всі категорії":
        filtered = filtered[filtered["category"] == selected_category]

    # розрахунок kpi
    total_rev = filtered["total"].sum()
    completed_orders = len(filtered[filtered["status"] == "complete"])
    aov = filtered["total"].mean() if len(filtered) > 0 else 0
    avg_discount = filtered["Discount_Percent"].mean() if len(filtered) > 0 else 0

    kpis = [
        html.Div(style={"backgroundColor": "#fff", "padding": "15px", "borderRadius": "8px", "textAlign": "center", "borderLeft": "5px solid #3498db"}, children=[
            html.H4("Загальна виручка", style={"margin": "0", "color": "#7f8c8d"}),
            html.H2(f"${total_rev / 1e6:.2f}M", style={"margin": "5px 0", "color": "#2c3e50"})
        ]),
        html.Div(style={"backgroundColor": "#fff", "padding": "15px", "borderRadius": "8px", "textAlign": "center", "borderLeft": "5px solid #2ecc71"}, children=[
            html.H4("Виконані замовлення", style={"margin": "0", "color": "#7f8c8d"}),
            html.H2(f"{completed_orders:,}", style={"margin": "5px 0", "color": "#2c3e50"})
        ]),
        html.Div(style={"backgroundColor": "#fff", "padding": "15px", "borderRadius": "8px", "textAlign": "center", "borderLeft": "5px solid #f39c12"}, children=[
            html.H4("Середній чек (AOV)", style={"margin": "0", "color": "#7f8c8d"}),
            html.H2(f"${aov:.2f}", style={"margin": "5px 0", "color": "#2c3e50"})
        ]),
        html.Div(style={"backgroundColor": "#fff", "padding": "15px", "borderRadius": "8px", "textAlign": "center", "borderLeft": "5px solid #e74c3c"}, children=[
            html.H4("Середня знижка", style={"margin": "0", "color": "#7f8c8d"}),
            html.H2(f"{avg_discount:.1f}%", style={"margin": "5px 0", "color": "#2c3e50"})
        ])
    ]

    # графік 1: щомісячна динаміка виручки та прибутку
    monthly_data = filtered.groupby("month_year").agg(revenue=("total", "sum"), profit=("profit", "sum")).reset_index()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=monthly_data["month_year"], y=monthly_data["revenue"] / 1e6, mode="lines+markers", name="Виручка (млн $)", line=dict(color="#3498db", width=3)))
    fig1.add_trace(go.Scatter(x=monthly_data["month_year"], y=monthly_data["profit"] / 1e6, mode="lines+markers", name="Прибуток (млн $)", line=dict(color="#2ecc71", width=3)))
    fig1.update_layout(title="Динаміка виручки та прибутку у часі", template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))

    # графік 2: структура продажів за категоріями з маржинальністю
    cat_data = filtered.groupby("category").agg(total_rev=("total", "sum"), total_prof=("profit", "sum")).reset_index()
    cat_data["margin_pct"] = (cat_data["total_prof"] / cat_data["total_rev"] * 100).fillna(0).round(1)
    cat_data = cat_data.sort_values("total_rev", ascending=True).tail(8)
    fig2 = px.bar(cat_data, x="total_rev", y="category", orientation="h", color="margin_pct",
                  color_continuous_scale="Viridis", labels={"total_rev": "Виручка ($)", "category": "Категорія", "margin_pct": "Маржа (%)"},
                  title="Структура продажів за категоріями та маржинальність")
    fig2.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))

    # графік 3: продажі за регіонами та статусами замовлень
    status_filter = filtered[filtered["status"].isin(["complete", "canceled", "received", "order_refunded"])]
    reg_stat = status_filter.groupby(["Region", "status"])["total"].sum().reset_index()
    fig3 = px.bar(reg_stat, x="Region", y="total", color="status", barmode="group",
                  labels={"total": "Сума ($)", "Region": "Регіон", "status": "Статус"},
                  title="Порівняння обсягів замовлень за регіонами та статусами")
    fig3.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))

    # графік 4: зв'язок ціни та знижки з розміром бульбашки за кількістю
    sample_scatter = filtered[filtered["total"] > 0].sample(n=min(300, len(filtered)), random_state=42)
    fig4 = px.scatter(sample_scatter, x="price", y="Discount_Percent", size="qty_ordered", color="category",
                      labels={"price": "Ціна за одиницю ($)", "Discount_Percent": "Знижка (%)", "qty_ordered": "Кількість"},
                      title="Взаємозв'язок ціни, знижки та обсягу замовлення")
    fig4.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=40, b=40))

    return kpis, fig1, fig2, fig3, fig4

if __name__ == "__main__":
    app.run(debug=True, port=8050)
