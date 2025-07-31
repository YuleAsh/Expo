#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import datetime

st.set_page_config(layout="wide")

# === Tenant-Specific Simulator ===
st.title("📍 Project Vantage LLC – Financial Yield Simulator")
st.markdown("A focused, finance-first simulator to assess ROI and yield potential for a single tenant.")

# === Lease & Financial Setup ===
st.header("📌 Basic Lease Details")
col1, col2, col3 = st.columns(3)
with col1:
    handover_date = st.date_input("Handover Date", datetime.date(2022, 11, 1))
    rent_start = st.date_input("Rent Commencement Date", datetime.date(2023, 12, 1))
with col2:
    lease_expiry = st.date_input("Lease Expiry Date", datetime.date(2028, 12, 1))
    rent_free_months = st.slider("Rent-Free Period (months)", 0, 24, 12)
with col3:
    payment_terms = st.selectbox("Payment Terms", ["Monthly", "Quarterly"], index=1)
    annual_rent_value = st.number_input("Annual Rent Value (AED)", 1000000, 10000000, 3500000, step=100000)

lease_duration_years = (lease_expiry - rent_start).days / 365
monthly_rent = annual_rent_value / 12
months_total = lease_duration_years * 12
months_rentable = months_total - rent_free_months

total_rent = monthly_rent * months_rentable

# === Land & Investment View ===
st.header("🏗️ Land Investment & ROI")
col1, col2, col3 = st.columns(3)
with col1:
    land_cost = st.number_input("Land Acquisition Cost (AED)", 1000000, 10000000, 3000000, step=500000)
    construction_cost = st.number_input("Construction Cost (AED)", 1000000, 10000000, 5000000, step=500000)
with col2:
    other_dev_costs = st.number_input("Other Development Costs (AED)", 0, 5000000, 1000000, step=250000)
    target_roi = st.slider("Target ROI (%)", 0.0, 20.0, 10.0)
with col3:
    land_appreciation_rate = st.slider("Land Appreciation (% per year)", 0.0, 15.0, 5.0)
    appreciation_horizon = st.slider("Appreciation Horizon (years)", 1, 20, int(lease_duration_years))

# === Calculations ===
total_investment = land_cost + construction_cost + other_dev_costs
land_appreciation_value = land_cost * ((1 + land_appreciation_rate / 100) ** appreciation_horizon)
expected_roi_value = total_investment * (target_roi / 100)

# === Financial Summary ===
st.header("📊 Financial Summary")
st.metric("Total Rent (AED)", f"{total_rent:,.0f}")
st.metric("Total Investment (AED)", f"{total_investment:,.0f}")
st.metric("Target ROI Value (AED)", f"{expected_roi_value:,.0f}")
st.metric("Land Value in {appreciation_horizon} yrs", f"{land_appreciation_value:,.0f}")

net_profit = total_rent - total_investment
roi_achieved = (net_profit / total_investment) * 100

st.subheader("💡 ROI vs Target")
st.write(f"**Achieved ROI:** {roi_achieved:.2f}%")
if roi_achieved >= target_roi:
    st.success("✅ On track to meet or exceed target ROI")
else:
    st.warning("⚠️ Below target ROI — consider optimizing rent or reducing costs")

# === Yield Table ===
st.subheader("📋 Yield Components")
yield_df = pd.DataFrame({
    "Component": [
        "Total Rent (AED)", "Total Investment (AED)", "Net Profit (AED)",
        "Target ROI (%)", "Achieved ROI (%)", "Appreciated Land Value (AED)"
    ],
    "Value": [
        f"{total_rent:,.0f}", f"{total_investment:,.0f}", f"{net_profit:,.0f}",
        f"{target_roi:.2f}%", f"{roi_achieved:.2f}%", f"{land_appreciation_value:,.0f}"
    ]
})
st.dataframe(yield_df, use_container_width=True)

st.caption("*All numbers are for simulation and illustrative purposes only.*")

