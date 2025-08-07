
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Retail Lease Simulator - V13", layout="wide")
st.title("🏬 Retail Store Lease Profitability Simulator (v2)")

# Sidebar Inputs
with st.sidebar:
    st.header("📌 Lease & Financial Assumptions")
    plot_area_sqm = st.number_input("Plot Area (sqm)", value=298.59, step=10.0)
    plot_area_sqft = round(plot_area_sqm * 10.7639, 2)
    lease_years = 6
    base_rent = st.number_input("Annual Rent (د.إ)", value=449960)
    rent_escalation = st.slider("Rent Escalation (%/yr)", 0.0, 10.0, 2.5)
    service_charge = st.number_input("Service Charges (د.إ/sqft)", value=27.5)
    rent_free_months = st.slider("Rent-Free Period (months)", 0, 12, 4)

    st.markdown("---")
    st.header("💰 Revenue Assumptions")
    uae_avg_revenue = st.slider("UAE Avg Revenue (د.إ)", 20_000_00, 80_000_00, 64_000_00, step=1_000_00)
    revenue_percent = st.slider("Store Revenue (% of Avg)", 40, 80, 60, step=5)
    store_revenue = revenue_percent / 100 * uae_avg_revenue

    st.markdown("---")
    st.header("💸 Annual Cost Components")
    cost_components = {
        "Electricity": st.slider("Electricity", 0, 150000, 70492, step=1000),
        "Water": st.slider("Water", 0, 30000, 12543, step=500),
        "Waste Mgmt": st.slider("Waste Mgmt", 0, 30000, 16070, step=500),
        "Facilities Mgmt": st.slider("Facilities Mgmt", 0, 150000, 88385, step=1000),
        "Gas": st.slider("Gas", 0, 100000, 56872, step=1000),
    }

# Backend Calculations
lease_start = datetime(2024, 8, 1)
years = [2024 + i for i in range(lease_years)]
months_in_year = [5, 12, 12, 12, 12, 7]
service_total = service_charge * plot_area_sqft
total_cost = sum(cost_components.values())
cost_schedule = [round(total_cost * (m / 12)) for m in months_in_year]
escalation_rate = rent_escalation / 100

base_rent_schedule = []
service_schedule = []
revenue_schedule = []
profit_schedule = []
margin_schedule = []
total_cost_schedule = []

for i, months in enumerate(months_in_year):
    paid_months = max(0, months - rent_free_months) if i == 0 else months
    annual_rent_i = base_rent * ((1 + escalation_rate) ** i)
    rent = annual_rent_i * (paid_months / 12)
    service = service_total * (paid_months / 12)
    revenue = store_revenue * (months / 12)
    cost = cost_schedule[i]
    profit = revenue - (rent + service + cost)
    margin = profit / revenue if revenue != 0 else 0

    base_rent_schedule.append(round(rent))
    service_schedule.append(round(service))
    total_cost_schedule.append(round(cost))
    revenue_schedule.append(round(revenue))
    profit_schedule.append(round(profit))
    margin_schedule.append(round(margin * 100, 1))

# KPI Cards
total_profit = sum(profit_schedule)
total_revenue = sum(revenue_schedule)
total_costs = sum(total_cost_schedule)
lease_roi = round(total_profit / total_costs * 100, 2) if total_costs else 0
avg_margin = np.mean(margin_schedule)

col1, col2, col3 = st.columns(3)
col1.metric("💼 Total Profit", f"د.إ {total_profit:,.0f}")
col2.metric("📊 Avg Profit Margin", f"{avg_margin:.1f}%")
col3.metric("🪙 Total ROI", f"{lease_roi:.1f}%")

# Financial Table
st.markdown("### 📘 Financial Year-wise Breakdown")
df = pd.DataFrame({
    "Financial Year": years,
    "Revenue (د.إ)": revenue_schedule,
    "Base Rent (د.إ)": base_rent_schedule,
    "Service Charges (د.إ)": service_schedule,
    "Other Costs (د.إ)": total_cost_schedule,
    "Profit (د.إ)": profit_schedule,
    "Margin (%)": margin_schedule
})

# Format and align the table
df_display = df.copy()
currency_cols = ["Revenue (د.إ)", "Base Rent (د.إ)", "Service Charges (د.إ)", "Other Costs (د.إ)", "Profit (د.إ)"]
for col in currency_cols:
    df_display[col] = df_display[col].apply(lambda x: f"د.إ {int(x):,}")
df_display["Margin (%)"] = df_display["Margin (%)"].apply(lambda x: f"{x:.1f}%")

st.markdown("""
    <style>
    table {
        width: 100% !important;
        font-size: 16px;
    }
    th, td {
        text-align: center !important;
        padding: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown(df_display.to_html(index=False), unsafe_allow_html=True)

# Excel Download
st.download_button(
    label="📥 Download Financial Table as Excel",
    data=df.to_csv(index=False),
    file_name="financial_breakdown.csv",
    mime="text/csv"
)

# Profit Bridge Chart
st.markdown("### 💧 Executive Profit Bridge")
bridge_fig = go.Figure()
bridge_fig.add_trace(go.Waterfall(
    name="Profit Bridge",
    orientation="v",
    measure=["absolute", "relative", "relative", "relative", "total"],
    x=["Revenue", "Base Rent", "Service", "Other Costs", "Net Profit"],
    y=[total_revenue, -sum(base_rent_schedule), -sum(service_schedule), -total_costs, total_profit],
    connector={"line": {"color": "gray"}}
))
bridge_fig.update_layout(height=400, title="Revenue to Profit Bridge")
st.plotly_chart(bridge_fig, use_container_width=True)

# Margin Trend and Cost Chart
st.markdown("### 📈 Margin Trend & Cost Comparison")
margin_fig = go.Figure()
margin_fig.add_trace(go.Bar(x=years, y=base_rent_schedule, name="Base Rent", yaxis='y'))
margin_fig.add_trace(go.Bar(x=years, y=service_schedule, name="Service Charges", yaxis='y'))
margin_fig.add_trace(go.Bar(x=years, y=total_cost_schedule, name="Other Costs", yaxis='y'))

margin_fig.add_trace(go.Scatter(
    x=years, y=margin_schedule,
    name="Profit Margin (%)", yaxis='y2',
    mode='lines+markers+text',
    text=[f"{m:.1f}%" for m in margin_schedule],
    textposition="top center",
    textfont=dict(size=12, color='black', family='Arial', weight='bold')
))

margin_fig.update_layout(
    barmode='stack',
    yaxis=dict(title="Costs (د.إ)"),
    yaxis2=dict(title="Margin %", overlaying='y', side='right', range=[0, max(margin_schedule) + 10]),
    title="Cost Breakdown vs Profit Margin Trend",
    height=450
)
st.plotly_chart(margin_fig, use_container_width=True)

# Cost Composition Chart
st.markdown("### 🔍 Cost Composition Overview")
sorted_costs = sorted(cost_components.items(), key=lambda x: x[1], reverse=True)
cost_bar = go.Figure(go.Bar(
    x=[v for _, v in sorted_costs],
    y=[k for k, _ in sorted_costs],
    orientation='h',
    marker=dict(color='mediumseagreen')
))
cost_bar.update_layout(height=400, title="Annual Cost Components", xaxis_title="د.إ", yaxis_title="")
st.plotly_chart(cost_bar, use_container_width=True)
