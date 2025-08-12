
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from copy import deepcopy

# ------------------------------
# Config / Assumptions
# ------------------------------
CAP_RATE = 0.085  # 8.5% synthetic cap rate for property value estimation

# ------------------------------
# Tenant Presets (3 Tenants)
# ------------------------------
TENANT_PRESETS = {
    "Tenant A (Original)": {
        "plot_area_sqm": 298.59,
        "lease_years": 6,
        "base_rent": 449_960,
        "rent_escalation": 2.5,  # %
        "service_charge": 27.5,  # per sqft
        "rent_free_months": 4,
        "uae_avg_revenue": 6_400_000,
        "revenue_percent": 60,   # % of UAE Avg Revenue
        "cost_components": {
            "Electricity": 70_492,
            "Water": 12_543,
            "Waste Mgmt": 16_070,
            "Facilities Mgmt": 88_385,
            "Gas": 56_872,
        },
    },
    "Tenant B (Synthetic)": {
        "plot_area_sqm": 320.0,
        "lease_years": 6,
        "base_rent": 475_000,
        "rent_escalation": 3.0,
        "service_charge": 29.0,
        "rent_free_months": 3,
        "uae_avg_revenue": 6_800_000,
        "revenue_percent": 62,
        "cost_components": {
            "Electricity": 76_000,
            "Water": 13_200,
            "Waste Mgmt": 17_500,
            "Facilities Mgmt": 92_000,
            "Gas": 60_000,
        },
    },
    "Tenant C (Synthetic)": {
        "plot_area_sqm": 270.0,
        "lease_years": 6,
        "base_rent": 430_000,
        "rent_escalation": 2.0,
        "service_charge": 31.0,
        "rent_free_months": 5,
        "uae_avg_revenue": 6_100_000,
        "revenue_percent": 58,
        "cost_components": {
            "Electricity": 66_000,
            "Water": 11_800,
            "Waste Mgmt": 15_500,
            "Facilities Mgmt": 85_000,
            "Gas": 54_000,
        },
    },
}

# ------------------------------
# Session State Initialization
# ------------------------------
if "tenant_state" not in st.session_state:
    st.session_state["tenant_state"] = deepcopy(TENANT_PRESETS)

# ------------------------------
# Core Financial Calculator
# ------------------------------
def compute_financials(cfg):
    """Compute year-wise schedules and KPIs for a tenant config dict."""
    plot_area_sqm = float(cfg["plot_area_sqm"])
    plot_area_sqft = round(plot_area_sqm * 10.7639, 2)
    lease_years = int(cfg["lease_years"])
    base_rent = float(cfg["base_rent"])
    rent_escalation = float(cfg["rent_escalation"])
    service_charge = float(cfg["service_charge"])
    rent_free_months = int(cfg["rent_free_months"])
    uae_avg_revenue = float(cfg["uae_avg_revenue"])
    revenue_percent = float(cfg["revenue_percent"])
    cost_components = cfg["cost_components"]

    years = [2024 + i for i in range(lease_years)]
    months_in_year = [5, 12, 12, 12, 12, 7]  # fixed as per v13 timeline
    service_total = service_charge * plot_area_sqft
    total_cost = float(sum(cost_components.values()))
    cost_schedule = [round(total_cost * (m / 12)) for m in months_in_year]
    escalation_rate = rent_escalation / 100.0
    store_revenue = revenue_percent / 100.0 * uae_avg_revenue

    base_rent_schedule, service_schedule = [], []
    revenue_schedule, profit_schedule = [], []
    margin_schedule, total_cost_schedule = [], []

    for i, months in enumerate(months_in_year):
        paid_months = max(0, months - rent_free_months) if i == 0 else months

        annual_rent_i = base_rent * ((1 + escalation_rate) ** i)
        rent = annual_rent_i * (paid_months / 12.0)
        service = service_total * (paid_months / 12.0)

        # --- Revenue now also respects paid_months in Year 1 ---
        revenue_months = paid_months if i == 0 else months
        revenue = store_revenue * (revenue_months / 12.0)

        cost = cost_schedule[i]
        profit = revenue - (rent + service + cost)
        margin = (profit / revenue) if revenue else 0.0

        base_rent_schedule.append(round(rent))
        service_schedule.append(round(service))
        total_cost_schedule.append(round(cost))
        revenue_schedule.append(round(revenue))
        profit_schedule.append(round(profit))
        margin_schedule.append(round(margin * 100.0, 1))

    # KPIs
    total_profit = sum(profit_schedule)
    total_revenue = sum(revenue_schedule)
    total_rent = sum(base_rent_schedule)
    total_service = sum(service_schedule)
    total_costs = sum(total_cost_schedule)
    total_expenses = total_service + total_costs
    lease_roi = round((total_profit / total_costs) * 100.0, 2) if total_costs else 0.0
    avg_margin = float(np.mean(margin_schedule)) if margin_schedule else 0.0

    df = pd.DataFrame({
        "Financial Year": years,
        "Revenue (AED)": revenue_schedule,
        "Base Rent (AED)": base_rent_schedule,
        "Service Charges (AED)": service_schedule,
        "Other Costs (AED)": total_cost_schedule,
        "Profit (AED)": profit_schedule,
    })
    df["Margin (%)"] = margin_schedule
    df["Cumulative Profit (AED)"] = np.cumsum(df["Profit (AED)"]).tolist()

    # Extremes & Payback
    peak_year = int(df.loc[df["Profit (AED)"].idxmax(), "Financial Year"])
    worst_year = int(df.loc[df["Profit (AED)"].idxmin(), "Financial Year"])
    payback_year = next((int(y) for y, cp in zip(df["Financial Year"], df["Cumulative Profit (AED)"]) if cp >= 0), None)

    # Synthetic Property Value (based on Year-1 base rent and cap rate)
    property_value = base_rent / CAP_RATE if CAP_RATE > 0 else 0.0
    total_yield_pct = (total_profit / property_value * 100.0) if property_value else 0.0

    kpis = dict(
        total_profit=total_profit,
        total_revenue=total_revenue,
        total_rent=total_rent,
        total_service=total_service,
        total_costs=total_costs,
        total_expenses=total_expenses,
        lease_roi=lease_roi,
        avg_margin=avg_margin,
        peak_year=peak_year,
        worst_year=worst_year,
        payback_year=payback_year,
        property_value=property_value,
        total_yield_pct=total_yield_pct
    )

    return years, df, kpis

# ------------------------------
# Page Layout & Sidebar
# ------------------------------
st.set_page_config(page_title="Retail Lease Simulator — v13 (Exec Summary Enhancements)", layout="wide")
st.title("🏬 Retail Store Lease Profitability Simulator — v13")

with st.sidebar:
    st.header("📌 Lease & Financial Assumptions")
    tenant_names = list(st.session_state["tenant_state"].keys())
    tenant_name = st.selectbox("Select Tenant", tenant_names, index=0)
    cfg = st.session_state["tenant_state"][tenant_name]

    # Inputs (keys are unique per tenant via f"..._{tenant_name}")
    plot_area_sqm = st.number_input("Plot Area (sqm)", value=float(cfg["plot_area_sqm"]), step=10.0, key=f"plot_area_{tenant_name}")
    lease_years = int(cfg["lease_years"])  # fixed per v13 (no widget)
    base_rent = st.number_input("Annual Rent (AED)", value=float(cfg["base_rent"]), step=1000.0, key=f"base_rent_{tenant_name}")
    rent_escalation = st.slider("Rent Escalation (%/yr)", 0.0, 10.0, float(cfg["rent_escalation"]), key=f"rent_esc_{tenant_name}")
    service_charge = st.number_input("Service Charges (AED/sqft)", value=float(cfg["service_charge"]), step=0.5, key=f"service_{tenant_name}")
    rent_free_months = st.slider("Rent-Free Period (months)", 0, 12, int(cfg["rent_free_months"]), key=f"rent_free_{tenant_name}")

    st.markdown("---")
    st.header("💰 Revenue Assumptions")
    uae_avg_revenue = st.slider("UAE Avg Revenue (AED)", 20_000_00, 80_000_00, int(cfg["uae_avg_revenue"]), step=1_000_00, key=f"uae_avg_rev_{tenant_name}")
    revenue_percent = st.slider("Store Revenue (% of Avg)", 40, 80, int(cfg["revenue_percent"]), step=5, key=f"rev_pct_{tenant_name}")

    st.markdown("---")
    st.header("💸 Annual Cost Components (AED)")
    elec = st.slider("Electricity", 0, 150000, int(cfg["cost_components"]["Electricity"]), step=1000, key=f"elec_{tenant_name}")
    water = st.slider("Water", 0, 30000, int(cfg["cost_components"]["Water"]), step=500, key=f"water_{tenant_name}")
    waste = st.slider("Waste Mgmt", 0, 30000, int(cfg["cost_components"]["Waste Mgmt"]), step=500, key=f"waste_{tenant_name}")
    facil = st.slider("Facilities Mgmt", 0, 150000, int(cfg["cost_components"]["Facilities Mgmt"]), step=1000, key=f"facil_{tenant_name}")
    gas = st.slider("Gas", 0, 100000, int(cfg["cost_components"]["Gas"]), step=1000, key=f"gas_{tenant_name}")

    # Persist current tenant's values
    st.session_state["tenant_state"][tenant_name] = {
        "plot_area_sqm": plot_area_sqm,
        "lease_years": lease_years,
        "base_rent": base_rent,
        "rent_escalation": rent_escalation,
        "service_charge": service_charge,
        "rent_free_months": rent_free_months,
        "uae_avg_revenue": uae_avg_revenue,
        "revenue_percent": revenue_percent,
        "cost_components": {
            "Electricity": elec, "Water": water, "Waste Mgmt": waste, "Facilities Mgmt": facil, "Gas": gas
        }
    }

    # ------------------------------
    # 🧩 Global Apply (restored)
    # ------------------------------
    st.markdown("---")
    st.subheader("🧩 Global Apply")
    st.caption("Use these to broadcast settings to all tenants in one click.")

    global_rf = st.slider("Set Rent-Free Period for ALL tenants (months)", 0, 12, rent_free_months, key="global_rent_free_slider")
    if st.button("Apply Rent-Free to ALL Tenants"):
        for tn in st.session_state["tenant_state"].keys():
            st.session_state["tenant_state"][tn]["rent_free_months"] = int(global_rf)
        st.success(f"Applied Rent-Free = {global_rf} months to ALL tenants.")

    if st.button("Copy ALL current tenant settings to ALL tenants"):
        src = deepcopy(st.session_state["tenant_state"][tenant_name])
        for tn in st.session_state["tenant_state"].keys():
            st.session_state["tenant_state"][tn] = deepcopy(src)
        st.success(f"Copied ALL settings from '{tenant_name}' to ALL tenants.")

# ------------------------------
# Compute per-tenant results & portfolio
# ------------------------------
tenant_results = {}
for tname, tcfg in st.session_state["tenant_state"].items():
    years, df_t, kpis_t = compute_financials(tcfg)
    tenant_results[tname] = {"years": years, "df": df_t, "kpis": kpis_t}

# Selected tenant
curr = tenant_results[tenant_name]
df = curr["df"]
k = curr["kpis"]
years = curr["years"]

# Portfolio: sum year-wise across tenants
lease_years = 6
portfolio_df = pd.DataFrame({"Financial Year": [2024 + i for i in range(lease_years)]})
for col in ["Revenue (AED)", "Base Rent (AED)", "Service Charges (AED)", "Other Costs (AED)", "Profit (AED)"]:
    series_sum = np.zeros(lease_years, dtype=float)
    for t in tenant_results.values():
        s = t["df"][col].values
        series_sum[:len(s)] += s
    portfolio_df[col] = series_sum

portfolio_df["Margin (%)"] = np.where(portfolio_df["Revenue (AED)"] > 0,
                                      (portfolio_df["Profit (AED)"] / portfolio_df["Revenue (AED)"]) * 100.0, 0.0)
portfolio_df["Cumulative Profit (AED)"] = np.cumsum(portfolio_df["Profit (AED)"]).tolist()

# Portfolio KPIs
pt_total_profit = portfolio_df["Profit (AED)"].sum()
pt_total_revenue = portfolio_df["Revenue (AED)"].sum()
pt_total_rent = portfolio_df["Base Rent (AED)"].sum()
pt_total_service = portfolio_df["Service Charges (AED)"].sum()
pt_total_costs = portfolio_df["Other Costs (AED)"].sum()
pt_total_expenses = pt_total_service + pt_total_costs
pt_roi = round((pt_total_profit / pt_total_costs) * 100.0, 2) if pt_total_costs else 0.0
pt_avg_margin = (portfolio_df["Profit (AED)"].sum() / pt_total_revenue * 100.0) if pt_total_revenue else 0.0
pt_peak_year = int(portfolio_df.loc[portfolio_df["Profit (AED)"].idxmax(), "Financial Year"])
pt_worst_year = int(portfolio_df.loc[portfolio_df["Profit (AED)"].idxmin(), "Financial Year"])
pt_payback_year = next((int(y) for y, cp in zip(portfolio_df["Financial Year"], portfolio_df["Cumulative Profit (AED)"]) if cp >= 0), None)

# Synthetic property value for portfolio
pt_property_value = sum(res["kpis"]["property_value"] for res in tenant_results.values())
pt_total_yield_pct = (pt_total_profit / pt_property_value * 100.0) if pt_property_value else 0.0

# ------------------------------
# Tabs
# ------------------------------
tab_dash, tab_exec = st.tabs(["📊 Dashboard (Selected Tenant)", "🧾 Executive Summary"])

with tab_dash:
    # KPI Cards (selected tenant)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Profit (AED)", f"{k['total_profit']:,.0f}")
    col2.metric("Avg Profit Margin", f"{k['avg_margin']:.1f}%")
    col3.metric("ROI on Costs", f"{k['lease_roi']:.1f}%")

    # Financial Table (selected tenant)
    st.markdown("### 📘 Financial Year-wise Breakdown — Selected Tenant (AED)")

    df_display = df.copy()
    currency_cols = ["Revenue (AED)", "Base Rent (AED)", "Service Charges (AED)", "Other Costs (AED)", "Profit (AED)", "Cumulative Profit (AED)"]
    for col in currency_cols:
        df_display[col] = df_display[col].apply(lambda x: f"{int(x):,}")
    df_display["Margin (%)"] = df_display["Margin (%)"].apply(lambda x: f"{x:.1f}%")

    st.markdown("""
        <style>
        table { width: 100% !important; font-size: 16px; }
        th, td { text-align: center !important; padding: 8px !important; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(df_display.to_html(index=False), unsafe_allow_html=True)

    # Download CSV (selected tenant)
    st.download_button(
        label="📥 Download Financial Table (Selected Tenant)",
        data=df.to_csv(index=False),
        file_name=f"financial_breakdown_{tenant_name.replace(' ', '_').replace('(', '').replace(')', '')}.csv",
        mime="text/csv"
    )

    # Profit Bridge (selected tenant)
    st.markdown("### 💧 Executive Profit Bridge — Selected Tenant")
    bridge_fig = go.Figure()
    bridge_fig.add_trace(go.Waterfall(
        name="Profit Bridge",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Revenue", "Base Rent", "Service", "Other Costs", "Net Profit"],
        y=[k["total_revenue"], -k["total_rent"], -k["total_service"], -k["total_costs"], k["total_profit"]],
        connector={"line": {"color": "gray"}}
    ))
    bridge_fig.update_layout(height=400, title="Revenue to Profit Bridge (Selected Tenant)")
    st.plotly_chart(bridge_fig, use_container_width=True)

    # Margin Trend & Cost (selected tenant)
    st.markdown("### 📈 Margin Trend & Cost Comparison — Selected Tenant")
    margin_fig = go.Figure()
    margin_fig.add_trace(go.Bar(x=years, y=df["Base Rent (AED)"], name="Base Rent", yaxis='y'))
    margin_fig.add_trace(go.Bar(x=years, y=df["Service Charges (AED)"], name="Service Charges", yaxis='y'))
    margin_fig.add_trace(go.Bar(x=years, y=df["Other Costs (AED)"], name="Other Costs", yaxis='y'))
    margin_fig.add_trace(go.Scatter(
        x=years, y=df["Margin (%)"],
        name="Profit Margin (%)", yaxis='y2',
        mode='lines+markers+text',
        text=[f"{m:.1f}%" for m in df["Margin (%)"]],
        textposition="top center",
    ))
    margin_fig.update_layout(
        barmode='stack',
        yaxis=dict(title="Costs (AED)"),
        yaxis2=dict(title="Margin %", overlaying='y', side='right', range=[0, max(df['Margin (%)']) + 10]),
        height=450
    )
    st.plotly_chart(margin_fig, use_container_width=True)

    # Cost composition (selected tenant)
    st.markdown("### 🔍 Cost Composition Overview — Selected Tenant")
    curr_cfg = st.session_state["tenant_state"][tenant_name]
    curr_costs = curr_cfg["cost_components"]
    sorted_costs = sorted(curr_costs.items(), key=lambda x: x[1], reverse=True)
    cost_bar = go.Figure(go.Bar(
        x=[v for _, v in sorted_costs],
        y=[k_ for k_, _ in sorted_costs],
        orientation='h'
    ))
    cost_bar.update_layout(height=400, title="Annual Cost Components (AED)", xaxis_title="AED", yaxis_title="")
    st.plotly_chart(cost_bar, use_container_width=True)

with tab_exec:
    scope = st.radio("Summary Scope", ["Portfolio (All Tenants)", "Selected Tenant"], index=0, horizontal=True)

    # Helper: combined YoY vs Cumulative chart with labels on cumulative
    def yoy_cumulative_chart(df_in, title="YoY Profit & Cumulative Profit"):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_in["Financial Year"], y=df_in["Profit (AED)"], name="Profit (AED)", yaxis="y"))
        fig.add_trace(go.Scatter(
            x=df_in["Financial Year"],
            y=df_in["Cumulative Profit (AED)"],
            name="Cumulative Profit (AED)",
            mode="lines+markers+text",
            text=[f"{int(v):,}" for v in df_in["Cumulative Profit (AED)"]],
            textposition="top center",
            textfont=dict(color="black", size=13),  # bold-like: larger size, black
            yaxis="y2"
        ))
        fig.update_layout(
            height=420,
            title=title,
            yaxis=dict(title="Profit (AED)"),
            yaxis2=dict(title="Cumulative (AED)", overlaying='y', side='right')
        )
        return fig

    st.caption(f"Property value is estimated synthetically using a {CAP_RATE*100:.1f}% cap rate applied to Year-1 base rent. 'Total Yield' is Total Profit ÷ Property Value.")

    if scope == "Selected Tenant":
        st.subheader("Executive Summary — Selected Tenant")

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue (AED)", f"{k['total_revenue']:,.0f}")
        c2.metric("Total Expenses (AED)", f"{k['total_expenses']:,.0f}")
        c3.metric("Total Profit (AED)", f"{k['total_profit']:,.0f}")
        c4.metric("Avg Margin", f"{k['avg_margin']:.1f}%")

        c5, c6, c7 = st.columns(3)
        c5.metric("ROI on Costs", f"{k['lease_roi']:.1f}%")
        c6.metric("Property Value (AED)", f"{k['property_value']:,.0f}")
        c7.metric("Total Yield (%)", f"{k['total_yield_pct']:.2f}%")

        # YoY + Cumulative visual
        st.markdown("---")
        st.subheader("Profit Progression")
        st.plotly_chart(yoy_cumulative_chart(df, "YoY Profit & Cumulative — Selected Tenant"), use_container_width=True)

        # Cash flow table & download
        st.markdown("---")
        st.subheader("Cash Flow View — Selected Tenant (AED)")
        cash_df = df[["Financial Year", "Revenue (AED)", "Base Rent (AED)", "Service Charges (AED)", "Other Costs (AED)", "Profit (AED)", "Cumulative Profit (AED)", "Margin (%)"]].copy()
        st.dataframe(cash_df, use_container_width=True)
        st.download_button(
            "📥 Download Executive Summary (Selected Tenant)",
            data=cash_df.to_csv(index=False),
            file_name=f"executive_summary_selected_{tenant_name.replace(' ', '_')}.csv",
            mime="text/csv"
        )

    else:
        st.subheader("Executive Summary — Portfolio (All Tenants)")

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue (AED)", f"{pt_total_revenue:,.0f}")
        c2.metric("Total Expenses (AED)", f"{pt_total_expenses:,.0f}")
        c3.metric("Total Profit (AED)", f"{pt_total_profit:,.0f}")
        c4.metric("Avg Margin (Weighted)", f"{pt_avg_margin:.1f}%")

        c5, c6, c7 = st.columns(3)
        c5.metric("ROI on Costs", f"{pt_roi:.1f}%")
        c6.metric("Property Value (AED)", f"{pt_property_value:,.0f}")
        c7.metric("Total Yield (%)", f"{pt_total_yield_pct:.2f}%")

        # YoY + Cumulative visual
        st.markdown("---")
        st.subheader("Portfolio Profit Progression")
        st.plotly_chart(yoy_cumulative_chart(portfolio_df, "YoY Profit & Cumulative — Portfolio"), use_container_width=True)

        # Per-tenant snapshot
        st.markdown("---")
        st.subheader("Per-Tenant KPI Snapshot")
        snap_rows = []
        for tname, res in tenant_results.items():
            kx = res["kpis"]
            snap_rows.append({
                "Tenant": tname,
                "Total Revenue (AED)": kx["total_revenue"],
                "Total Expenses (AED)": kx["total_expenses"],
                "Total Profit (AED)": kx["total_profit"],
                "Avg Margin (%)": kx["avg_margin"],
                "ROI (%)": kx["lease_roi"],
                "Property Value (AED)": kx["property_value"],
                "Total Yield (%)": kx["total_yield_pct"],
                "Peak Year": kx["peak_year"],
                "Worst Year": kx["worst_year"],
                "Payback Year": kx["payback_year"] if kx["payback_year"] is not None else "—",
            })
        snap_df = pd.DataFrame(snap_rows)
        st.dataframe(snap_df, use_container_width=True)

        # Portfolio CSV
        st.download_button(
            "📥 Download Portfolio Summary (CSV)",
            data=portfolio_df.to_csv(index=False),
            file_name="portfolio_executive_summary.csv",
            mime="text/csv"
        )
