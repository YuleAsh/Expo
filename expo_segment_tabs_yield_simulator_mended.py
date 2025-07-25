
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("Expo City Yield Optimizer - Segment-Specific Simulator")

tabs = st.tabs(["🏢 Office", "🛍️ Retail", "🍽️ F&B"])

# Shared calculation function
def calculate_yield(params):
    base_rent, occupancy, rent_free, lease_years, area, cooling, smart_util, leasing_cost_pct, churn, anchor_ratio = params
    revenue = base_rent * occupancy * (1 - rent_free / (lease_years * 12)) * area
    cost = cooling * area * (1 - smart_util * 0.2)
    leasing_cost = leasing_cost_pct * revenue * (1 + churn * 0.5) * (1 - anchor_ratio * 0.5)
    return (revenue - cost - leasing_cost) / area

# ================== 🏢 Office Tab ====================
with tabs[0]:
    st.header("🏢 Office Yield Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        base_rent = st.slider("Base Rent (AED/sqm)", 50, 300, 180, step=5, key="office_base_rent")
        occupancy = st.slider("Occupancy Rate (%)", 60, 100, 85, step=1, key="office_occupancy") / 100
        lease_years = st.slider("Lease Term (yrs)", 1, 10, 5, key="office_lease_term")
    with col2:
        rent_free = st.slider("Rent-Free Months", 0, 12, 2, key="office_rent_free")
        area = st.number_input("Leased Area (sqm)", 1000, 100000, 20000, step=1000, key="office_area")
        cooling = st.slider("Cooling Cost (AED/sqm/month)", 5, 40, 15, key="office_cooling")
    with col3:
        smart_util = st.slider("Smart Utility Uptake (%)", 0, 100, 50, key="office_smart_util") / 100
        leasing_cost_pct = st.slider("Leasing Cost (% of Rev)", 0, 20, 5, key="office_leasing_cost") / 100
        churn = st.slider("Tenant Churn (%)", 0, 50, 10, key="office_churn") / 100
        anchor_ratio = st.slider("Anchor Tenants (%)", 0, 100, 25, key="office_anchor") / 100

    st.subheader("🏢 Office-Specific Parameters")
    col1, col2 = st.columns(2)
    with col1:
        parking_ratio = st.slider("Parking Ratio (spaces/100sqm)", 0, 10, 4, key="office_parking")
        meeting_room_density = st.slider("Meeting Rooms per 1000sqm", 0, 10, 3, key="office_meeting_rooms")
    with col2:
        internet_uptime = st.slider("Avg Internet Uptime (%)", 80, 100, 98, key="office_internet")
        avg_staff_density = st.slider("Staff Density (per 100sqm)", 2, 10, 6, key="office_staff_density")
        green_cert = st.selectbox("Green Certification", ["None", "LEED Silver", "LEED Gold", "LEED Platinum"], key="office_green_cert")

    net_yield = calculate_yield([base_rent, occupancy, rent_free, lease_years, area, cooling, smart_util,
                                 leasing_cost_pct, churn, anchor_ratio])
    st.metric("🏢 Net Yield (AED/sqm)", f"{net_yield:.2f}")

# ================== 🛍️ Retail Tab ====================
with tabs[1]:
    st.header("🛍️ Retail Yield Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        base_rent = st.slider("Base Rent (AED/sqm)", 100, 500, 300, step=10, key="retail_base_rent")
        occupancy = st.slider("Occupancy Rate (%)", 50, 100, 90, step=1, key="retail_occupancy") / 100
        lease_years = st.slider("Lease Duration (yrs)", 1, 10, 4, key="retail_lease_term")
    with col2:
        rent_free = st.slider("Rent-Free Months", 0, 12, 1, key="retail_rent_free")
        area = st.number_input("Retail Area (sqm)", 500, 50000, 10000, step=500, key="retail_area")
        cooling = st.slider("Cooling Cost (AED/sqm/month)", 5, 50, 18, key="retail_cooling")
    with col3:
        smart_util = st.slider("Smart Utility Uptake (%)", 0, 100, 60, key="retail_smart_util") / 100
        leasing_cost_pct = st.slider("Leasing Cost (% of Rev)", 0, 25, 6, key="retail_leasing_cost") / 100
        churn = st.slider("Retail Churn Rate (%)", 0, 60, 20, key="retail_churn") / 100
        anchor_ratio = st.slider("Anchor Retailers (%)", 0, 100, 35, key="retail_anchor") / 100

    st.subheader("🛍️ Retail-Specific Parameters")
    col1, col2 = st.columns(2)
    with col1:
        avg_footfall = st.slider("Monthly Footfall (000s)", 0, 500, 120, key="retail_footfall")
        fnb_mix_pct = st.slider("F&B % in Retail Mix", 0, 100, 30, key="retail_fnb_mix")
    with col2:
        promo_days = st.slider("Promo Days/Month", 0, 31, 4, key="retail_promo_days")
        signage_quality = st.selectbox("Signage Quality", ["Low", "Average", "High"], key="retail_signage")
        avg_ticket_size = st.slider("Avg Purchase (AED)", 20, 500, 120, key="retail_ticket_size")

    net_yield = calculate_yield([base_rent, occupancy, rent_free, lease_years, area, cooling, smart_util,
                                 leasing_cost_pct, churn, anchor_ratio])
    st.metric("🛍️ Net Yield (AED/sqm)", f"{net_yield:.2f}")

# ================== 🍽️ F&B Tab ====================
with tabs[2]:
    st.header("🍽️ F&B Yield Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        base_rent = st.slider("Base Rent (AED/sqm)", 100, 450, 250, step=10, key="fnb_base_rent")
        occupancy = st.slider("Occupancy Rate (%)", 50, 100, 88, key="fnb_occupancy") / 100
        lease_years = st.slider("Lease Term (yrs)", 1, 10, 3, key="fnb_lease_term")
    with col2:
        rent_free = st.slider("Rent-Free Months", 0, 12, 1, key="fnb_rent_free")
        area = st.number_input("F&B Space (sqm)", 200, 20000, 5000, step=200, key="fnb_area")
        cooling = st.slider("Cooling Cost (AED/sqm/month)", 5, 60, 25, key="fnb_cooling")
    with col3:
        smart_util = st.slider("Smart Meter Uptake (%)", 0, 100, 70, key="fnb_smart_util") / 100
        leasing_cost_pct = st.slider("Leasing/Admin Cost (% of Rev)", 0, 30, 8, key="fnb_leasing_cost") / 100
        churn = st.slider("F&B Churn Rate (%)", 0, 60, 25, key="fnb_churn") / 100
        anchor_ratio = st.slider("Flagship %", 0, 100, 40, key="fnb_anchor") / 100

    st.subheader("🍽️ F&B-Specific Parameters")
    col1, col2 = st.columns(2)
    with col1:
        avg_table_turn = st.slider("Avg Table Turns/Day", 1, 10, 4, key="fnb_turns")
        delivery_ratio = st.slider("Delivery Orders %", 0, 100, 30, key="fnb_delivery")
    with col2:
        kitchen_area_ratio = st.slider("Kitchen as % of Unit", 0, 100, 40, key="fnb_kitchen")
        waste_mgmt_score = st.selectbox("Waste Management", ["Poor", "Average", "Excellent"], key="fnb_waste")
        avg_fnb_ticket = st.slider("Avg Spend (AED)", 20, 500, 80, key="fnb_ticket")

    net_yield = calculate_yield([base_rent, occupancy, rent_free, lease_years, area, cooling, smart_util,
                                 leasing_cost_pct, churn, anchor_ratio])
    st.metric("🍽️ Net Yield (AED/sqm)", f"{net_yield:.2f}")
