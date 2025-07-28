#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import datetime

st.set_page_config(layout="wide")

# === Tenant-Specific Simulator ===
st.title("📍 Project Vantage LLC – Tailored Yield Simulator")
st.markdown("Simulate financial outcomes for Project Vantage LLC using real lease parameters, cost assumptions, and what-if toggles.")

# === Lease Dates & Timeline ===
st.header("📅 Lease Timeline")
handover_date = st.date_input("Handover Date", datetime.date(2022, 11, 1))
rent_start = st.date_input("Rent Commencement Date", datetime.date(2023, 12, 1))
lease_expiry = st.date_input("Lease Expiry Date", datetime.date(2028, 12, 1))

rent_free_months = st.slider("Rent-Free Period (months)", 0, 24, 12)
lease_duration_years = (lease_expiry - rent_start).days / 365

# === Lease & Area Details ===
st.header("🏢 Lease Details")
annual_rent_value = st.number_input("Annual Rent Value (AED)", 500000, 10000000, 3500000, step=100000)
rent_per_sqm = st.number_input("Rent Price per sqm (AED)", 100, 1000, 500, step=50)
internal_area = st.number_input("Internal Net Leasable Area (sqm)", 100, 1000, 300, step=50)
external_area = st.number_input("External Net Leasable Area (sqm)", 0, 1000, 100, step=50)
total_area = internal_area + external_area

# === Variable Controls ===
st.header("⚙️ Operating Costs & Modifiers")
buggy_cost_month = st.number_input("Buggy Services Cost (AED/month)", 0, 10000, 2000, step=500)
building_repair = st.slider("Building Repairs (AED/sqm/month)", 0, 100, 20)
cleaning_cost = st.slider("Cleaning Cost (AED/sqm/month)", 0, 100, 10)
maintenance_cost = st.slider("Maintenance Cost (AED/sqm/month)", 0, 100, 15)
insurance_cost = st.slider("Insurance (% of Annual Rent)", 0.0, 10.0, 1.5)
late_fee_rate = st.slider("Late Payment Fee (% of Quarterly Payment)", 0.0, 10.0, 2.0)
late_payment_toggle = st.toggle("Simulate Late Payment?", value=True)

# === Other Variables ===
st.header("💳 Other Terms")
payment_terms = st.selectbox("Payment Terms", ["Monthly", "Quarterly", "Bi-Annually"], index=1)
security_deposit_pct = st.slider("Security Deposit (% of Annual Rent)", 0, 20, 10)
rent_review_cap = st.slider("Rent Review Cap (%)", 0, 20, 10)

# === Optional Charges ===
parking_spaces = total_area // 50
parking_cost_per_space = st.slider("Parking Cost per Space (AED/month)", 0, 2000, 500)
charge_for_parking = st.toggle("Charge for Parking?", value=False)
service_charges = st.number_input("Service Charges (AED/month)", 0, 100000, 0, step=1000)
master_community_fee = st.number_input("Master Community Fee (AED/month)", 0, 100000, 0, step=1000)

# === Calculations ===
months = lease_duration_years * 12
months_rentable = months - rent_free_months
monthly_rent = annual_rent_value / 12

# Cost calculations
total_buggy = buggy_cost_month * months
total_repair = building_repair * total_area * months
total_cleaning = cleaning_cost * total_area * months
total_maintenance = maintenance_cost * total_area * months
total_insurance = (insurance_cost / 100) * annual_rent_value * lease_duration_years
total_late_fee = (monthly_rent * (3 if payment_terms == "Quarterly" else 1) * (late_fee_rate / 100)) if late_payment_toggle else 0
total_parking = parking_spaces * parking_cost_per_space * months if charge_for_parking else 0
total_service = service_charges * months
total_master_fee = master_community_fee * months

# Revenue
total_rent = monthly_rent * months_rentable

total_costs = sum([
    total_buggy,
    total_repair,
    total_cleaning,
    total_maintenance,
    total_insurance,
    total_late_fee,
    total_parking,
    total_service,
    total_master_fee
])

net_yield_aed = (total_rent - total_costs) / total_area / lease_duration_years

# === Output ===
st.header("📊 Yield Summary")
st.metric("Net Yield (AED/sqm/year)", f"{net_yield_aed:,.2f}")

st.subheader("📋 Breakdown")
st.dataframe(pd.DataFrame({
    "Component": [
        "Total Rent Collected", "Buggy Services", "Repairs", "Cleaning", "Maintenance",
        "Insurance", "Late Fees", "Parking Revenue", "Service Charges", "Community Fee",
        "Total Costs", "Net Yield (AED/sqm/year)"
    ],
    "Value (AED)": [
        f"{total_rent:,.0f}", f"{total_buggy:,.0f}", f"{total_repair:,.0f}", f"{total_cleaning:,.0f}",
        f"{total_maintenance:,.0f}", f"{total_insurance:,.0f}", f"{total_late_fee:,.0f}",
        f"{total_parking:,.0f}", f"{total_service:,.0f}", f"{total_master_fee:,.0f}",
        f"{total_costs:,.0f}", f"{net_yield_aed:,.2f}"
    ]
}))

st.caption("*All calculations are estimates based on input assumptions. For demonstration only.*")

