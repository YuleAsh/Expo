#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd

# ✨ Title
st.set_page_config(layout="wide")
st.title("Expo City Yield Optimization Simulator")
st.markdown("Use the controls below to simulate Net Yield across different leasing scenarios.")

# ⬇️ Three-column layout for input sections
col1, col2, col3 = st.columns(3)

# === Column 1: Revenue Drivers ===
with col1.expander("📈 Revenue Parameters", expanded=True):
    base_rent = st.slider("Base Rent per sqm (AED)", 50, 250, 150)
    occupancy_rate = st.slider("Occupancy Rate (%)", 50, 100, 85) / 100
    rent_free_months = st.slider("Rent-Free Period (months)", 0, 12, 2)
    lease_duration = st.slider("Lease Duration (years)", 1, 10, 5)
    leased_area = st.number_input("Total Leased Area (sqm)", 1000, 100000, 20000, step=1000)

# === Column 2: Operating Costs ===
with col2.expander("💡 Operating Cost Inputs", expanded=True):
    cooling_cost = st.slider("Cooling Cost/sqm (AED/month)", 5, 50, 15)
    smart_util_pct = st.slider("Smart Utility Uptake (%)", 0, 100, 50) / 100
    metro_footfall = st.slider("Metro Footfall (000s/day)", 0, 50, 8)

# === Column 3: Leasing/Admin & External Modifiers ===
with col3.expander("📉 Leasing/Admin Parameters", expanded=True):
    leasing_cost_pct = st.slider("Leasing Admin Cost (% of Revenue)", 0, 20, 5) / 100
    tenant_churn = st.slider("Tenant Churn Rate (%)", 0, 50, 15) / 100
    anchor_ratio = st.slider("Anchor Tenant Ratio (%)", 0, 100, 20) / 100

with col3.expander("🌍 External Modifiers", expanded=True):
    gdp_growth = st.slider("GDP Growth (%)", -5.0, 10.0, 3.0)
    inflation = st.slider("Inflation Rate (%)", 0.0, 15.0, 4.0)
    tourism = st.slider("Tourism Arrivals (000s/month)", 0, 500, 120)
    retail_index = st.slider("Retail Sales Index (baseline = 100)", 50, 200, 120)

# === Yield Calculation ===
st.header("📊 Yield Summary")

# Step 1: Base revenue
effective_revenue = base_rent * occupancy_rate * (1 - rent_free_months / (lease_duration * 12)) * leased_area

# Step 2: Cost side
operating_cost = cooling_cost * leased_area * (1 - smart_util_pct * 0.2)  # 20% cost saved
leasing_cost = leasing_cost_pct * effective_revenue * (1 + tenant_churn * 0.5) * (1 - anchor_ratio * 0.5)

# Step 3: External factor adjustment
modifier_effect = (
    (gdp_growth / 100) * 0.1
    - (inflation / 100) * 0.08
    + ((tourism - 100) / 1000) * 0.05
    + ((retail_index - 100) / 100) * 0.1
)

base_yield = (effective_revenue - operating_cost - leasing_cost) / leased_area
adjusted_yield = base_yield * (1 + modifier_effect)

# === Output Metrics ===
st.metric("🧮 Adjusted Net Yield (AED/sqm)", f"{adjusted_yield:,.2f}")

# === Breakdown Table ===
with st.expander("📑 Yield Components Breakdown", expanded=False):
    st.dataframe(pd.DataFrame({
        "Component": ["Effective Revenue", "Operating Cost", "Leasing/Admin Cost", "Base Yield", "Adjusted Yield"],
        "Value (AED)": [
            f"{effective_revenue:,.0f}",
            f"{operating_cost:,.0f}",
            f"{leasing_cost:,.0f}",
            f"{base_yield:,.2f} per sqm",
            f"{adjusted_yield:,.2f} per sqm"
        ]
    }))

# === Commentary ===
with st.expander("📢 Market Commentary", expanded=False):
    st.markdown(f"- **GDP Growth:** {gdp_growth:.1f}% → positive leasing signal")
    st.markdown(f"- **Inflation:** {inflation:.1f}% → increases OpEx burden")
    st.markdown(f"- **Tourism Arrivals:** {tourism:,}K/month → boosts retail & F&B")
    st.markdown(f"- **Retail Index:** {retail_index} → reflects demand trend")

