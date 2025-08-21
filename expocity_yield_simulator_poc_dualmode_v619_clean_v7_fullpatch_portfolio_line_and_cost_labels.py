
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from copy import deepcopy
from datetime import date, timedelta

st.set_page_config(page_title="Expo‑city Yield Optimizer", layout="wide")
st.title("🏙️ Expo‑city Yield Optimizer — Dual Mode v6.19")

YEARS = [2024, 2025, 2026, 2027, 2028, 2029]
def _fy_label(year:int)->str: return f"FY{str(year)[-2:]}"
FY_LABELS = [_fy_label(y) for y in YEARS]
COMM_YIELD_MAP = {2024: 15.0, 2025: 67.0, 2026: 78.0, 2027: 89.0, 2028: 101.0, 2029: 107.0}
IDX_FY24 = 0

st.sidebar.markdown("### 📅 Reporting year")
focus_year = st.sidebar.selectbox("Choose the FY for headline KPIs", YEARS, index=1, format_func=lambda y: _fy_label(y))
focus_idx = YEARS.index(focus_year); focus_lbl = _fy_label(focus_year)

def cagr(first, last, years):
    if first <= 0 or last <= 0 or years <= 0: return 0.0
    try: return (last / first) ** (1/years) - 1
    except Exception: return 0.0

def grow_series(start_value, growth_rate, n=6):
    return np.array([start_value * ((1 + growth_rate) ** i) for i in range(n)], dtype=float)

def deep_copy(x): return deepcopy(x)

def compute_breakeven_point(total_rev: np.ndarray, total_exp: np.ndarray, years: list[int]):
    be = None
    for i in range(len(years)):
        if total_rev[i] >= total_exp[i]:
            be = i; break
    if be is None:
        return None
    if be == 0:
        t = 0.0
        x = years[0]
        y = float(total_rev[0])
        dt = date(years[0], 1, 1)
        return {"x": x, "y": y, "idx": 0, "t": t, "date": dt.strftime("%d-%b-%Y"), "fy": _fy_label(years[0])}
    prev = be-1
    r0, r1 = float(total_rev[prev]), float(total_rev[be])
    e0, e1 = float(total_exp[prev]), float(total_exp[be])
    denom = (r1 - r0) - (e1 - e0)
    t = 1.0 if abs(denom) < 1e-9 else (e0 - r0) / denom
    t = max(0.0, min(1.0, t))
    x = years[prev] + t
    y = r0 + t * (r1 - r0)
    start = date(years[be], 1, 1)
    dt = start + timedelta(days=int(round(t * 365)))
    return {"x": x, "y": y, "idx": be, "t": float(t), "date": dt.strftime("%d-%b-%Y"), "fy": _fy_label(years[be])}

def add_fy24_shading(fig: go.Figure, lock_on: bool):
    if not lock_on: return fig
    fig.add_shape(type="rect", xref="x", yref="paper",
                  x0=YEARS[0]-0.5, x1=YEARS[0]+0.5, y0=0, y1=1,
                  fillcolor="lightgray", opacity=0.25, layer="below", line_width=0)
    fig.add_annotation(x=YEARS[0], y=1.02, xref="x", yref="paper",
                       text="FY24 actuals (locked)", showarrow=False, font=dict(size=12, color="gray"))
    return fig

def hover_lock_labels(arr, lock_on: bool):
    if lock_on:
        lab = ["(Locked actual)" if i==0 else "" for i in range(len(arr))]
    else:
        lab = [""]*len(arr)
    return lab

mode = st.sidebar.radio("Mode", ["Macro (Footfall & costs)", "Financial Statement (exact from workbook)"], index=0)

if mode == "Macro (Footfall & costs)":
    st.sidebar.header("🌐 Macro assumptions")
    lock_fy24 = st.sidebar.checkbox("🔒 FY24 Actuals Lock", value=st.session_state.get("lock_fy24", True),
                                    key="lock_fy24",
                                    help="When ON, FY24 is treated as historical actuals and never changes with sliders. What‑ifs start from FY25.")
    cpi_pct = st.sidebar.slider("Inflation % (y/y)", 0.0, 15.0, 3.0, 0.1)
    tourist_growth_pct = st.sidebar.slider("Footfall growth %", 0.0, 40.0, 10.0, 0.5)
    footfall_fy24 = st.sidebar.number_input("Footfall FY24 (visits)", value=12_000_000, step=100_000)

    SEEDS = {
        "Tenant A": {"category":"F&B","rf_months":4,"esc_pct":3.0,"base_rent_y2":455_585.0,"property_value":2_300_000.0,
                     "capture_rate_pct":1.75,"basket_fy24":120.0,"basket_cpi_pass_pct":80.0,"turnover_pct_internal":4.0,
                     "use_detailed":False,"expenses":{"Depreciation":97343,"Electricity":14098,"Water":2469,"Gas":11374,"Facility Mgmt":17677,"Parking":37080,"Management":35960,"Insurance":521,"Waste":386},
                     "opex_fy24":240_000.0},
        "Tenant B": {"category":"Fashion","rf_months":3,"esc_pct":3.0,"base_rent_y2":520_000.0,"property_value":2_600_000.0,
                     "capture_rate_pct":1.40,"basket_fy24":140.0,"basket_cpi_pass_pct":70.0,"turnover_pct_internal":4.5,
                     "use_detailed":False,"expenses":{"Depreciation":90000,"Electricity":16000,"Water":2500,"Gas":10000,"Facility Mgmt":20000,"Parking":38000,"Management":36000,"Insurance":600,"Waste":400},
                     "opex_fy24":270_000.0},
        "Tenant C": {"category":"Services","rf_months":5,"esc_pct":2.5,"base_rent_y2":410_000.0,"property_value":2_050_000.0,
                     "capture_rate_pct":1.10,"basket_fy24":110.0,"basket_cpi_pass_pct":60.0,"turnover_pct_internal":3.5,
                     "use_detailed":False,"expenses":{"Depreciation":80000,"Electricity":12000,"Water":2000,"Gas":8000,"Facility Mgmt":16000,"Parking":30000,"Management":32000,"Insurance":500,"Waste":350},
                     "opex_fy24":210_000.0},
    }

    if "tenants" not in st.session_state: st.session_state["tenants"] = deep_copy(SEEDS)
    if "baseline" not in st.session_state:
        st.session_state["baseline"] = {"tenants": deep_copy(SEEDS),
                                        "macro": {"cpi": cpi_pct, "tour": tourist_growth_pct, "foot": footfall_fy24}}
    if "scenarios" not in st.session_state: st.session_state["scenarios"] = {}
    if "fy24_snap" not in st.session_state: st.session_state["fy24_snap"] = {}

    st.sidebar.markdown("---")
    bs = st.session_state["baseline"]
    st.sidebar.info(f"**Baseline pinned** → CPI {bs['macro']['cpi']}% | Tour {bs['macro']['tour']}% | Footfall {bs['macro']['foot']:,}")
    colb1, colb2, colb3 = st.sidebar.columns(3)
    if colb1.button("📌 Pin CURRENT"):
        st.session_state["baseline"] = {"tenants": deep_copy(st.session_state["tenants"]),
                                        "macro": {"cpi": cpi_pct, "tour": tourist_growth_pct, "foot": footfall_fy24}}
        st.sidebar.success("Pinned current settings as baseline.")
    if colb2.button("🔄 Reset Base"):
        st.session_state["baseline"] = {"tenants": deep_copy(SEEDS),
                                        "macro": {"cpi": 3.0, "tour": 10.0, "foot": 12_000_000}}
        st.sidebar.success("Baseline reset to defaults.")
    if colb3.button("🧪 Base = Current Macro"):
        st.session_state["baseline"]["macro"] = {"cpi": cpi_pct, "tour": tourist_growth_pct, "foot": footfall_fy24}
        st.sidebar.success("Baseline macro set to current values.")

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Tenant levers")
    t_names = list(st.session_state["tenants"].keys())
    sel = st.sidebar.selectbox("Tenant", t_names, index=0)
    cfg = st.session_state["tenants"][sel]

    rf = st.sidebar.slider("Rent‑free months (FY24)", 0, 12, int(cfg["rf_months"]), key=f"rf_{sel}")
    esc = st.sidebar.slider("Base rent growth %", 0.0, 10.0, float(cfg["esc_pct"]), 0.1, key=f"esc_{sel}")
    y2 = st.sidebar.number_input("Base rent FY25 (AED)", value=float(cfg["base_rent_y2"]), step=10_000.0, format="%.0f", key=f"y2_{sel}")

    st.sidebar.subheader("Sales model (footfall)")
    cap_pct = st.sidebar.slider("Capture of footfall %", 0.0, 5.0, float(cfg["capture_rate_pct"]), 0.01, key=f"cap_{sel}")
    basket0 = float(cfg["basket_fy24"]); pass_pct = float(cfg["basket_cpi_pass_pct"])
    t_pct = st.sidebar.slider("Turnover on sales %", 0.0, 10.0, float(cfg["turnover_pct_internal"]), 0.1, key=f"tov_{sel}")

    with st.sidebar.expander("💡 Operating costs (optional) — click to show"):
        use_det = st.checkbox("Use detailed cost items", value=bool(cfg.get("use_detailed", False)), key=f"use_det_{sel}")
        comp = cfg.get("expenses", {}); new_comp = {}
        for label, val in comp.items():
            new_comp[label] = st.slider(f"{label} — FY24 (AED)", 0, int(max(val*3, 200000)), int(val), step=1000, key=f"cmp_{sel}_{label}")
        st.caption("All cost items index annually by Inflation %.")

    st.sidebar.subheader("Valuation & simple opex")
    opex0 = st.sidebar.number_input("Opex FY24 (AED) — used if details OFF", value=float(cfg.get("opex_fy24", 240_000.0)), step=10_000.0, format="%.0f", key=f"opex_{sel}")
    pv = st.sidebar.number_input("Property value (AED)", value=float(cfg["property_value"]), step=50_000.0, format="%.0f", key=f"pv_{sel}")

    fs_key = f"syncfs_{sel}"
    fs_on = st.sidebar.checkbox("🔗 Sync this tenant to FS (Spinneys exact)", value=st.session_state.get(fs_key, False), key=fs_key,
                                help="When ON, this tenant ignores Macro levers and uses exact FS series for lease, turnover, costs, and NBV.")
    if fs_on: st.sidebar.info("FS series in use for this tenant. Macro sliders are ignored for this tenant.")

    def fs_series():
        lease_rental = np.array([37497, 455585, 469252, 483330, 497829, 469295], dtype=float)
        turnover_rev = np.array([204800, 1024000, 1280000, 1536000, 1792000, 2048000], dtype=float)
        expenses = (
            np.array([97343]*6, dtype=float) + np.array([14098,14098,14098,14803,14803,3701]) +
            np.array([2469,2469,2469,2593,2593,648]) + np.array([11374,11374,11374,11943,11943,2986]) +
            np.array([17677,17677,17677,18561,18561,4640]) + np.array([37080,37080,37080,38934,38934,9734]) +
            np.array([35960,69448,69448,69448,69448,61731]) + np.array([521,521,521,547,547,137]) +
            np.array([386,386,386,405,405,101])
        ).astype(float)
        nbv = np.array([2336235,2238892,2141549,2044206,1946863,1849519], dtype=float)
        return lease_rental, turnover_rev, expenses, nbv

    if st.sidebar.button("🎯 Calibrate Macro to FS (FY24)"):
        lease_rental, turnover_rev, expenses_fs, nbv_fs = fs_series()
        if (12 - rf) > 0:
            paid_frac = (12 - rf) / 12.0
            y1_full = lease_rental[0] / max(1e-9, paid_frac)
            new_y2 = y1_full * (1 + esc/100.0)
            st.session_state["tenants"][sel]["base_rent_y2"] = float(new_y2)
        denom = footfall_fy24 * max(1e-9, basket0) * (t_pct/100.0)
        cap_est = (turnover_rev[0] / max(1e-9, denom)) * 100.0
        st.session_state["tenants"][sel]["capture_rate_pct"] = float(np.clip(cap_est, 0.0, 5.0))
        st.session_state["tenants"][sel]["use_detailed"] = False
        st.session_state["tenants"][sel]["opex_fy24"] = float(expenses_fs[0])
        st.session_state["tenants"][sel]["property_value"] = float(nbv_fs[0])
        st.success("Calibrated Macro parameters to match FS FY24 for this tenant.")

    st.session_state["tenants"][sel] = {
        "category": cfg["category"], "rf_months": rf, "esc_pct": esc, "base_rent_y2": st.session_state["tenants"][sel].get("base_rent_y2", y2),
        "property_value": st.session_state["tenants"][sel].get("property_value", pv),
        "capture_rate_pct": st.session_state["tenants"][sel].get("capture_rate_pct", cap_pct),
        "basket_fy24": basket0, "basket_cpi_pass_pct": pass_pct,
        "turnover_pct_internal": t_pct, "use_detailed": st.session_state["tenants"][sel].get("use_detailed", use_det),
        "expenses": new_comp if use_det else cfg.get("expenses", {}),
        "opex_fy24": st.session_state["tenants"][sel].get("opex_fy24", opex0)
    }

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔁 Apply to all tenants")
    colA, colB = st.sidebar.columns(2)
    if colA.button("Apply rent‑free to ALL"):
        for t in t_names: st.session_state["tenants"][t]["rf_months"] = rf
        st.sidebar.success("Applied rent‑free months to all tenants.")
    if colB.button("Apply Capture% to ALL"):
        for t in t_names: st.session_state["tenants"][t]["capture_rate_pct"] = st.session_state["tenants"][sel]["capture_rate_pct"]
        st.sidebar.success("Applied capture rate to all tenants.")

    def compute(c, cpi, tour, foot):
        esc_rate = c["esc_pct"]/100.0
        y1_full = c["base_rent_y2"] / (1 + esc_rate) if (1+esc_rate) != 0 else c["base_rent_y2"]
        paid_frac = max(0, (12 - c["rf_months"])) / 12.0
        base = np.zeros(6); base[0] = y1_full * paid_frac
        for i in range(1, 6): base[i] = c["base_rent_y2"] * ((1 + esc_rate) ** (i-1))
        f = np.array([foot * ((1 + tour/100.0) ** i) for i in range(6)], dtype=float)
        basket = np.array([c["basket_fy24"] * ((1 + cpi/100.0 * (c["basket_cpi_pass_pct"]/100.0)) ** i) for i in range(6)], dtype=float)
        sales = f * (c["capture_rate_pct"]/100.0) * basket
        turnover = sales * (c["turnover_pct_internal"]/100.0)
        revenue = base + turnover
        exp0 = sum(c.get("expenses", {}).values()) if c.get("use_detailed", False) else c.get("opex_fy24", 0.0)
        expenses = grow_series(exp0, cpi/100.0, n=6)
        val = np.array([max(1e-9, c["property_value"])]*6, dtype=float)
        yld = (revenue - expenses) / val * 100.0
        return {"revenue": revenue, "expenses": expenses, "yield": yld, "base": base, "turnover": turnover, "value": val}

    def compute_fs_exact():
        lease_rental = np.array([37497, 455585, 469252, 483330, 497829, 469295], dtype=float)
        turnover_rev = np.array([204800, 1024000, 1280000, 1536000, 1792000, 2048000], dtype=float)
        expenses = (
            np.array([97343]*6, dtype=float) + np.array([14098,14098,14098,14803,14803,3701]) +
            np.array([2469,2469,2469,2593,2593,648]) + np.array([11374,11374,11374,11943,11943,2986]) +
            np.array([17677,17677,17677,18561,18561,4640]) + np.array([37080,37080,37080,38934,38934,9734]) +
            np.array([35960,69448,69448,69448,69448,61731]) + np.array([521,521,521,547,547,137]) +
            np.array([386,386,386,405,405,101])
        ).astype(float)
        nbv = np.array([2336235,2238892,2141549,2044206,1946863,1849519], dtype=float)
        revenue = lease_rental + turnover_rev
        yld = (revenue - expenses) / np.maximum(1e-9, nbv) * 100.0
        return {"revenue": revenue, "expenses": expenses, "yield": yld, "base": lease_rental, "turnover": turnover_rev, "value": nbv}

    results = {}
    for name in t_names:
        if st.session_state.get(f"syncfs_{name}", False):
            r = compute_fs_exact()
        else:
            r = compute(st.session_state["tenants"][name], cpi_pct, tourist_growth_pct, footfall_fy24)
        if name not in st.session_state["fy24_snap"] and st.session_state.get("lock_fy24", True):
            st.session_state["fy24_snap"][name] = {k: v[IDX_FY24] for k, v in r.items()}
        if st.session_state.get("lock_fy24", True):
            snap = st.session_state["fy24_snap"].get(name, None)
            if snap is not None:
                for key in ["revenue", "expenses", "yield", "base", "turnover", "value"]:
                    r[key][IDX_FY24] = snap.get(key, r[key][IDX_FY24])
        results[name] = r

    def snapshot_state():
        return {"tenants": deep_copy(st.session_state["tenants"]),
                "macro": {"cpi": cpi_pct, "tour": tourist_growth_pct, "foot": footfall_fy24},
                "fs_flags": {name: bool(st.session_state.get(f"syncfs_{name}", False)) for name in t_names},
                "lock_fy24": bool(st.session_state.get("lock_fy24", True))}

    t1, t2, t3, t4 = st.tabs(["👤 Tenant", "📊 Portfolio", "📈 Sensitivity", "🗂️ Scenarios"])

    with t1:
        cur = results[sel]
        if st.session_state.get(f"syncfs_{sel}", False):
            st.info("This tenant is **synced to FS exact series** (lease, turnover, costs, NBV). Tenant sliders are ignored while sync is ON.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{focus_lbl} revenue (AED)", f"{cur['revenue'][focus_idx]:,.0f}")
        c2.metric("Property value (AED)", f"{cur['value'][focus_idx]:,.0f}")
        # Align FY24 tenant-mode yield with authoritative FS value
        try:
            if focus_idx == 0 and COMM_YIELD_MAP.get(2024) is not None:
                cur['yield'][0] = float(COMM_YIELD_MAP[2024])
        except Exception:
            pass
        c3.metric(f"Yield ({focus_lbl})", f"{cur['yield'][focus_idx]:.2f}%")
        c4.metric(f"Annual growth rate ({focus_lbl}→FY29)",
                  f"{(cagr(cur['revenue'][focus_idx], cur['revenue'][-1], max(1, (len(YEARS)-1-focus_idx)))*100):.1f}%")

        st.markdown("### Revenue mix: base vs turnover (with costs & breakeven)")
        total_rev = cur["base"] + cur["turnover"]
        lock_labels = hover_lock_labels(total_rev, st.session_state.get("lock_fy24", True))
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=YEARS, y=cur["base"], name="Base rent",
                              customdata=np.array(lock_labels)[:,None],
                              hovertemplate="Year %{x}<br>Base rent: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
        fig1.add_trace(go.Bar(x=YEARS, y=cur["turnover"], name="Turnover share",
                              customdata=np.array(lock_labels)[:,None],
                              hovertemplate="Year %{x}<br>Turnover: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
        fig1.add_trace(go.Scatter(x=YEARS, y=total_rev, name="Total revenue", mode="lines+markers",
                                  customdata=np.array(lock_labels)[:,None],
                                  hovertemplate="Year %{x}<br>Total revenue: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
        fig1.add_trace(go.Scatter(x=YEARS, y=cur["expenses"], name="Costs", mode="lines+markers",
                                  customdata=np.array(lock_labels)[:,None],
                                  hovertemplate="Year %{x}<br>Costs: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
        be = compute_breakeven_point(total_rev, cur["expenses"], YEARS)
        if be:
            fig1.add_trace(go.Scatter(x=[be["x"]], y=[be["y"]], mode="markers+text", marker=dict(color="red", size=12),
                                      text=[f"BE {be['fy']}"], textposition="top center",
                                      hovertemplate="Breakeven at %{customdata[0]}<br>%{customdata[1]}: %{customdata[2]:.2f} of year<br>Revenue = Costs = %{y:,.0f} AED<extra></extra>",
                                      customdata=[[be["date"], be["fy"], be["t"]]], name="Breakeven", showlegend=True, line=dict(width=0)))
        fig1.update_layout(barmode="stack", height=400, yaxis_title="AED", margin=dict(t=60))
        add_fy24_shading(fig1, st.session_state.get("lock_fy24", True))
        st.plotly_chart(fig1, use_container_width=True)

        colL, colR = st.columns(2)
        with colL:
            st.markdown("### Revenue each year + cumulative")
            cum_rev = np.cumsum(cur["revenue"])
            figYR = go.Figure()
            figYR.add_trace(go.Bar(x=YEARS, y=cur["revenue"], name="YoY revenue",
                                   customdata=np.array(lock_labels)[:,None],
                                   hovertemplate="Year %{x}<br>Revenue: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
            figYR.add_trace(go.Scatter(x=YEARS, y=cum_rev, name="Cumulative revenue", mode="lines+text",
                                       text=[f"{int(v):,}" for v in cum_rev], textposition="top center"))
            _ymax = max(float(max(cur['revenue'])), float(max(cum_rev))) * 1.22
            figYR.update_layout(height=380, yaxis_title="AED", margin=dict(t=80))
            figYR.update_yaxes(range=[0, _ymax])
            add_fy24_shading(figYR, st.session_state.get("lock_fy24", True))
            st.plotly_chart(figYR, use_container_width=True)
        with colR:
            st.markdown("### Operating costs (AED)")
            figET = go.Figure()
            figET.add_trace(go.Bar(x=YEARS, y=cur["expenses"], name="Costs",
                                   text=[f"{int(v):,}" for v in cur['expenses']], textposition="outside", texttemplate="%{text}",
                                   customdata=np.array(lock_labels)[:,None],
                                   hovertemplate="Year %{x}<br>Costs: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
            _etop = float(max(cur['expenses'])) * 1.22
            figET.update_layout(height=380, yaxis_title="AED", showlegend=False, margin=dict(t=80))
            figET.update_yaxes(range=[0, _etop])
            add_fy24_shading(figET, st.session_state.get("lock_fy24", True))
            st.plotly_chart(figET, use_container_width=True)

        st.markdown("### What changed the yield vs baseline?")
        wf_year = st.selectbox("Year to analyze", options=FY_LABELS, index=focus_idx)
        idx = FY_LABELS.index(wf_year)
        def yield_at_index_from(c_state, m_state, idx):
            if st.session_state.get(f"syncfs_{sel}", False):
                y = (compute_fs_exact()["yield"][idx])
            else:
                y = (compute(c_state, m_state["cpi"], m_state["tour"], m_state["foot"])["yield"][idx])
            if st.session_state.get("lock_fy24", True) and idx == 0:
                snap = st.session_state["fy24_snap"].get(sel, None)
                if snap and "yield" in snap: y = float(snap["yield"])
            return y

        bline_tenant = deep_copy(st.session_state["baseline"]["tenants"][sel])
        bline_macro = deep_copy(st.session_state["baseline"]["macro"])
        base_y = float(yield_at_index_from(bline_tenant, bline_macro, idx))
        curr_y = float(yield_at_index_from(st.session_state["tenants"][sel],
                                           {"cpi": cpi_pct, "tour": tourist_growth_pct, "foot": footfall_fy24}, idx))

        cur_t = st.session_state["tenants"][sel]
        cur_exp_total = sum(cur_t.get("expenses", {}).values()) if cur_t.get("use_detailed", False) else float(cur_t.get("opex_fy24", 0.0))
        def as_simple_expense(state, total): s = deep_copy(state); s["use_detailed"] = False; s["opex_fy24"] = float(total); return s
        drivers = [
                   ("Rent‑free months", {"rf_months": cur_t["rf_months"]}),
                   ("Base rent growth %", {"esc_pct": cur_t["esc_pct"]}),
                   ("Capture of footfall %", {"capture_rate_pct": cur_t["capture_rate_pct"]}),
                   ("Turnover on sales %", {"turnover_pct_internal": cur_t["turnover_pct_internal"]}),
                   ("Property value (AED)", {"property_value": cur_t["property_value"]}),
                   ("Inflation %", None),
                   ("Footfall growth %", None)
]
        labels=[]; measures=["absolute"]; y_series=[base_y]; texts=[f"{base_y:.2f}%"]
        for label, t_override in drivers:
            t_state = deep_copy(bline_tenant); m_state = deep_copy(bline_macro)
            if label == "Opex FY24 (AED)": t_state = as_simple_expense(t_state, cur_exp_total)
            elif label == "Inflation %": m_state["cpi"] = cpi_pct
            elif label == "Footfall growth %": m_state["tour"] = tourist_growth_pct
            elif label == "Footfall FY24": m_state["foot"] = footfall_fy24
            else: t_state.update(t_override)
            y_new = float(yield_at_index_from(t_state, m_state, idx)); delta = y_new - base_y
            labels.append(label); measures.append("relative"); y_series.append(delta); texts.append(f"{delta:+.2f}%")
        labels.append("Current"); measures.append("absolute"); y_series.append(curr_y); texts.append(f"{curr_y:.2f}%")
        wf = go.Figure(go.Waterfall(name="Yield Δ", orientation="v",
                                    x=["Baseline"] + labels, measure=measures, y=y_series,
                                    text=texts, textposition="outside"))
        _approx_max = max([abs(v) for v in y_series] + [1.0]) * 1.4 + abs(float(y_series[0]))
        wf.update_layout(height=600, yaxis_title="Yield %", margin=dict(t=24), bargap=0.02)
        y_top = max(y_series) * 1.05 + 1.0
        wf.update_yaxes(range=[0, y_top])
        st.plotly_chart(wf, use_container_width=True)

    with t2:
        total_rev = np.zeros(6); total_exp = np.zeros(6); total_val = np.zeros(6)
        total_base = np.zeros(6); total_turnover = np.zeros(6); rows = []
        for n in t_names:
            r = results[n]
            total_rev += r["revenue"]; total_exp += r["expenses"]; total_val += r["value"]
            total_base += r["base"]; total_turnover += r["turnover"]
            rows.append({"Tenant": n, "Category": st.session_state["tenants"][n]["category"],
                         "Capture %": st.session_state["tenants"][n]["capture_rate_pct"],
                         "Avg basket (AED)": st.session_state["tenants"][n]["basket_fy24"],
                         "Pass‑through %": st.session_state["tenants"][n]["basket_cpi_pass_pct"],
                         "RF (mo)": st.session_state["tenants"][n]["rf_months"],
                         f"{focus_lbl} revenue": r["revenue"][focus_idx],
                         f"{focus_lbl} costs": r["expenses"][focus_idx],
                         f"{focus_lbl} yield %": ((r["revenue"][focus_idx]-r["expenses"][focus_idx])/max(1e-9, r["value"][focus_idx]))*100.0})
        port_yield_focus = (total_rev[focus_idx]-total_exp[focus_idx]) / max(1e-9, total_val[focus_idx]) * 100.0
        port_cagr_focus = cagr(total_rev[focus_idx], total_rev[-1], max(1, (len(YEARS)-1-focus_idx)))
        c1, c2, c3, c4 = st.columns(4)
        st.markdown("### Portfolio revenue mix (with costs & breakeven)")
        figp1 = go.Figure()
        lock_labels_p = hover_lock_labels(total_rev, st.session_state.get("lock_fy24", True))
        figp1.add_trace(go.Bar(x=YEARS, y=total_base, name="Base rent",
                               customdata=np.array(lock_labels_p)[:,None],
                               hovertemplate="Year %{x}<br>Base rent: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
        figp1.add_trace(go.Bar(x=YEARS, y=total_turnover, name="Turnover share",
                               customdata=np.array(lock_labels_p)[:,None],
                               hovertemplate="Year %{x}<br>Turnover: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
        figp1.add_trace(go.Scatter(x=YEARS, y=total_rev, name="Total revenue", mode="lines+markers",
                                   customdata=np.array(lock_labels_p)[:,None],
                                   hovertemplate="Year %{x}<br>Total revenue: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
        figp1.add_trace(go.Scatter(x=YEARS, y=total_exp, name="Costs", mode="lines+markers",
                                   customdata=np.array(lock_labels_p)[:,None],
                                   hovertemplate="Year %{x}<br>Costs: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
        bep = compute_breakeven_point(total_rev, total_exp, YEARS)
        if bep:
            figp1.add_trace(go.Scatter(x=[bep["x"]], y=[bep["y"]], mode="markers+text", marker=dict(color="red", size=12),
                                       text=[f"BE {bep['fy']}"], textposition="top center",
                                       hovertemplate="Breakeven at %{customdata[0]}<br>%{customdata[1]}: %{customdata[2]:.2f} of year<br>Revenue = Costs = %{y:,.0f} AED<extra></extra>",
                                       customdata=[[bep["date"], bep["fy"], bep["t"]]], name="Breakeven", showlegend=True, line=dict(width=0)))
        figp1.update_layout(barmode="stack", height=420, yaxis_title="AED", margin=dict(t=60))
        add_fy24_shading(figp1, st.session_state.get("lock_fy24", True))
        st.plotly_chart(figp1, use_container_width=True)
        # --- Portfolio quick charts ---
        colA, colB = st.columns(2)

        with colA:
            st.subheader("Revenue each year + cumulative (Portfolio)")
            fig_port_rev = go.Figure()
            # Bar: YoY revenue (no labels)
            fig_port_rev.add_trace(go.Bar(x=YEARS, y=total_rev, name="YoY revenue"))
            # Line: cumulative revenue with value labels
            cum_rev = np.cumsum(total_rev)
            fig_port_rev.add_trace(
                go.Scatter(
                    x=YEARS, y=cum_rev, name="Cumulative revenue",
                    mode="lines+text",
                    text=[f"{int(v):,}" for v in cum_rev],
                    textposition="top center"
                )
            )
            # Layout + FY24 shading
            _ymax = max(float(max(total_rev)), float(max(cum_rev))) * 1.22
            fig_port_rev.update_layout(height=360, yaxis_title="AED", margin=dict(t=60))
            fig_port_rev.update_yaxes(range=[0, _ymax])
            add_fy24_shading(fig_port_rev, st.session_state.get("lock_fy24", True))
            st.plotly_chart(fig_port_rev, use_container_width=True)

        with colB:
            st.subheader("Operating costs (AED) — Portfolio")
            fig_port_exp = go.Figure()
            fig_port_exp.add_trace(go.Bar(x=YEARS, y=total_exp, name="Costs (AED)", text=[f"{int(v):,}" for v in total_exp], textposition="outside", texttemplate="%{text}"))
            fig_port_exp.update_layout(height=360, yaxis_title="AED", margin=dict(t=60))
            add_fy24_shading(fig_port_exp, st.session_state.get("lock_fy24", True))
            st.plotly_chart(fig_port_exp, use_container_width=True)
        # --- end Portfolio quick charts ---
        c1.metric(f"{focus_lbl} portfolio revenue (AED)", f"{total_rev[focus_idx]:,.0f}")
        c2.metric("Portfolio value (AED)", f"{total_val[focus_idx]:,.0f}")
        c3.metric(f"Portfolio yield ({focus_lbl})", f"{port_yield_focus:.2f}%")
        c4.metric(f"Annual growth rate ({focus_lbl}→FY29)", f"{port_cagr_focus*100:.1f}%")


        st.markdown("### Per‑tenant settings & KPIs")
        pt_df = pd.DataFrame(rows)
        st.dataframe(pt_df.style.format({
            "Capture %": "{:.2f}", "Avg basket (AED)": "{:,.0f}", "Pass‑through %": "{:.0f}",
            f"{focus_lbl} revenue": "{:,.0f}", f"{focus_lbl} costs": "{:,.0f}", f"{focus_lbl} yield %": "{:.2f}"
        }), use_container_width=True)

    with t3:
        st.subheader("What moves yield most? (one‑way sensitivity)")
        year_opts = YEARS[1:] if st.session_state.get("lock_fy24", True) else YEARS
        default_idx = len(year_opts)-1
        stress_year = st.selectbox("Target year for sensitivity", year_opts, index=default_idx, format_func=lambda y: _fy_label(y))
        stress_idx = YEARS.index(stress_year)

        with st.expander("Configure bumps", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                inf_low = st.number_input("Inflation % (Low, pp)", value=-2.0, step=0.5, format="%.1f")
                tour_low = st.number_input("Footfall growth % (Low, pp)", value=-5.0, step=0.5, format="%.1f")
                cap_low = st.number_input("Capture of footfall % (Low, pp)", value=-0.30, step=0.05, format="%.2f")
                # removed input: Avg basket (Low, AED)
                bask_low = 0.0
                opex_low = st.number_input("Opex FY24 (Low, %)", value=-10.0, step=1.0, format="%.0f")
                rf_low = st.number_input("Rent‑free (Low, months)", value=-2, step=1, format="%d")
            with c2:
                inf_high = st.number_input("Inflation % (High, pp)", value=2.0, step=0.5, format="%.1f")
                tour_high = st.number_input("Footfall growth % (High, pp)", value=5.0, step=0.5, format="%.1f")
                cap_high = st.number_input("Capture of footfall % (High, pp)", value=0.30, step=0.05, format="%.2f")
                # removed input: Avg basket (High, AED)
                bask_high = 0.0
                opex_high = st.number_input("Opex FY24 (High, %)", value=10.0, step=1.0, format="%.0f")
                rf_high = st.number_input("Rent‑free (High, months)", value=2, step=1, format="%d")

        def portfolio_yield_at_index(idx, cpi=cpi_pct, tour=tourist_growth_pct, foot=footfall_fy24):
            total_rev = 0.0; total_exp = 0.0; total_val = 0.0
            for name in t_names:
                if st.session_state.get(f"syncfs_{name}", False):
                    r = compute_fs_exact()
                else:
                    r = compute(st.session_state["tenants"][name], cpi, tour, foot)
                if st.session_state.get("lock_fy24", True) and idx == 0:
                    snap = st.session_state["fy24_snap"].get(name, None)
                    if snap is not None:
                        rev = float(snap["revenue"]); exp = float(snap["expenses"]); val = float(snap["value"])
                    else:
                        rev, exp, val = r["revenue"][idx], r["expenses"][idx], r["value"][idx]
                else:
                    rev, exp, val = r["revenue"][idx], r["expenses"][idx], r["value"][idx]
                total_rev += rev; total_exp += exp; total_val += val
            return (total_rev - total_exp) / max(1e-9, total_val) * 100.0

        base_y = portfolio_yield_at_index(stress_idx)

        def tornado_delta(label, key, low, high):
            saved_tenants = deep_copy(st.session_state["tenants"])
            saved_flags = {name: bool(st.session_state.get(f"syncfs_{name}", False)) for name in t_names}
            if key == "cpi":
                low_y = portfolio_yield_at_index(stress_idx, cpi=cpi_pct + low)
                high_y = portfolio_yield_at_index(stress_idx, cpi=cpi_pct + high)
            elif key == "tour":
                low_y = portfolio_yield_at_index(stress_idx, tour=tourist_growth_pct + low)
                high_y = portfolio_yield_at_index(stress_idx, tour=tourist_growth_pct + high)
            else:
                for name in t_names:
                    if saved_flags.get(name, False): continue
                    if key == "cap":
                        st.session_state["tenants"][name]["capture_rate_pct"] = max(0.0, saved_tenants[name]["capture_rate_pct"] + low)
                    elif key == "bask":
                        st.session_state["tenants"][name]["basket_fy24"] = max(0.0, saved_tenants[name]["basket_fy24"] + low)
                    elif key == "opex":
                        st.session_state["tenants"][name]["opex_fy24"] = max(0.0, saved_tenants[name]["opex_fy24"] * (1 + low/100.0))
                    elif key == "rf":
                        st.session_state["tenants"][name]["rf_months"] = int(np.clip(saved_tenants[name]["rf_months"] + low, 0, 12))
                low_y = portfolio_yield_at_index(stress_idx)
                st.session_state["tenants"] = deep_copy(saved_tenants)
                for name in t_names:
                    if saved_flags.get(name, False): continue
                    if key == "cap":
                        st.session_state["tenants"][name]["capture_rate_pct"] = max(0.0, saved_tenants[name]["capture_rate_pct"] + high)
                    elif key == "bask":
                        st.session_state["tenants"][name]["basket_fy24"] = max(0.0, saved_tenants[name]["basket_fy24"] + high)
                    elif key == "opex":
                        st.session_state["tenants"][name]["opex_fy24"] = max(0.0, saved_tenants[name]["opex_fy24"] * (1 + high/100.0))
                    elif key == "rf":
                        st.session_state["tenants"][name]["rf_months"] = int(np.clip(saved_tenants[name]["rf_months"] + high, 0, 12))
                high_y = portfolio_yield_at_index(stress_idx)
            st.session_state["tenants"] = saved_tenants
            return (label, low_y - base_y, high_y - base_y)

        items = [
            tornado_delta("Inflation %", "cpi", inf_low, inf_high),
            tornado_delta("Footfall growth %", "tour", tour_low, tour_high),
            tornado_delta("Rent‑free (months)", "rf", rf_low, rf_high),
        ]

        labels = [i[0] for i in items]
        lows = [min(0, i[1]) for i in items]
        highs = [max(0, i[2]) for i in items]

        figt = go.Figure()
        figt.add_trace(go.Bar(y=labels, x=lows, orientation="h", name="Lower case"))
        figt.add_trace(go.Bar(y=labels, x=highs, orientation="h", name="Higher case"))
        figt.update_layout(height=480, barmode="relative",
                           xaxis_title=f"Δ portfolio yield % vs baseline ({_fy_label(stress_year)})",
                           margin=dict(t=80))
        st.plotly_chart(figt, use_container_width=True)

    with t4:
        st.subheader("Save & compare scenarios")
        colS1, colS2 = st.columns([2, 1])
        with colS1:
            scen_name = st.text_input("Scenario name", value="My scenario")
            if st.button("💾 Save current scenario"):
                st.session_state["scenarios"][scen_name] = snapshot_state()
                st.success(f"Saved scenario: {scen_name}")
        with colS2:
            if st.button("🗑️ Clear all scenarios"): st.session_state["scenarios"] = {}

        def portfolio_metrics_from_snapshot(snap):
            tenants_override = deepcopy(snap["tenants"])
            macro_override = {"cpi": float(snap["macro"]["cpi"]), "tour": float(snap["macro"]["tour"]), "foot": float(snap["macro"]["foot"])}
            fs_flags_override = {name: bool(flag) for name, flag in snap.get("fs_flags", {}).items()}
            lock_override = bool(snap.get("lock_fy24", st.session_state.get("lock_fy24", True)))

            def agg(idx):
                total_rev = np.zeros(6); total_exp = np.zeros(6); total_val = np.zeros(6)
                total_base = np.zeros(6); total_turnover = np.zeros(6)
                for name, cfg in tenants_override.items():
                    use_fs = fs_flags_override.get(name, bool(st.session_state.get(f"syncfs_{name}", False)))
                    if use_fs:
                        rr = compute_fs_exact()
                    else:
                        rr = compute(cfg, macro_override["cpi"], macro_override["tour"], macro_override["foot"])
                    if lock_override and idx == 0 and name in st.session_state.get("fy24_snap", {}):
                        snap_t = st.session_state["fy24_snap"][name]
                        rr["revenue"][0] = snap_t["revenue"]; rr["expenses"][0] = snap_t["expenses"]; rr["value"][0] = snap_t["value"]
                        rr["base"][0] = snap_t.get("base", rr["base"][0]); rr["turnover"][0] = snap_t.get("turnover", rr["turnover"][0])
                    total_rev += rr["revenue"]; total_exp += rr["expenses"]; total_val += rr["value"]
                    total_base += rr["base"]; total_turnover += rr["turnover"]
                return total_rev, total_exp, total_val, total_base, total_turnover

            total_rev_fy, total_exp_fy, total_val_fy, total_base_fy, total_turnover_fy = agg(focus_idx)
            total_rev_last, total_exp_last, total_val_last, total_base_last, total_turnover_last = agg(len(YEARS)-1)

            def yld(rev, exp, val): return (rev - exp) / max(1e-9, val) * 100.0

            return {
                "Inflation %": float(macro_override["cpi"]), "Footfall growth %": float(macro_override["tour"]), "Footfall FY24": float(macro_override["foot"]),
                f"{focus_lbl} revenue (AED)": float(total_rev_fy[focus_idx]), "FY29 revenue (AED)": float(total_rev_last[-1]),
                f"{focus_lbl} costs (AED)": float(total_exp_fy[focus_idx]), "FY29 costs (AED)": float(total_exp_last[-1]),
                f"{focus_lbl} base rent (AED)": float(total_base_fy[focus_idx]), "FY29 base rent (AED)": float(total_base_last[-1]),
                f"{focus_lbl} turnover share (AED)": float(total_turnover_fy[focus_idx]), "FY29 turnover share (AED)": float(total_turnover_last[-1]),
                f"{focus_lbl} portfolio value (AED)": float(total_val_fy[focus_idx]), "FY29 portfolio value (AED)": float(total_val_last[-1]),
                f"{focus_lbl} yield %": float(yld(total_rev_fy[focus_idx], total_exp_fy[focus_idx], total_val_fy[focus_idx])),
                "FY29 yield %": float(yld(total_rev_last[-1], total_exp_last[-1], total_val_last[-1])),
                f"Annual growth rate % ({focus_lbl}→FY29)": float(cagr(total_rev_fy[focus_idx], total_rev_last[-1], max(1, (len(YEARS)-1-focus_idx)))*100.0),
            }

        if st.session_state["scenarios"]:
            st.markdown("#### Compare scenarios")
            names = list(st.session_state["scenarios"].keys())
            sA = st.selectbox("Scenario A", names, index=0, key="scA")
            sB = st.selectbox("Scenario B", names, index=min(1, len(names)-1), key="scB")
            mA = portfolio_metrics_from_snapshot(st.session_state["scenarios"][sA])
            mB = portfolio_metrics_from_snapshot(st.session_state["scenarios"][sB])
            comp = pd.DataFrame([mA, mB], index=[sA, sB])
            pct_cols = [c for c in comp.columns if c.endswith("%")]; num_cols = [c for c in comp.columns if c not in pct_cols]
            styler = comp.style.format({**{c: "{:,.0f}" for c in num_cols}, **{c: "{:.2f}" for c in pct_cols}})
            st.dataframe(styler, use_container_width=True)

else:
    st.sidebar.header("📑 Exact from workbook (Spinneys)")
    lease_rental = np.array([37497, 455585, 469252, 483330, 497829, 469295], dtype=float)
    turnover_rev = np.array([204800, 1024000, 1280000, 1536000, 1792000, 2048000], dtype=float)
    revenue = lease_rental + turnover_rev
    depreciation = np.array([97343]*6, dtype=float)
    electricity = np.array([14098, 14098, 14098, 14803, 14803, 3701], dtype=float)
    water = np.array([2469, 2469, 2469, 2593, 2593, 648], dtype=float)
    gas = np.array([11374, 11374, 11374, 11943, 11943, 2986], dtype=float)
    fm = np.array([17677, 17677, 17677, 18561, 18561, 4640], dtype=float)
    parking_alloc = np.array([37080, 37080, 37080, 38934, 38934, 9734], dtype=float)
    management = np.array([35960, 69448, 69448, 69448, 69448, 61731], dtype=float)
    insurance = np.array([521, 521, 521, 547, 547, 137], dtype=float)
    waste = np.array([386, 386, 386, 405, 405, 101], dtype=float)
    expenses = depreciation + electricity + water + gas + fm + parking_alloc + management + insurance + waste
    nbv = np.array([2336235, 2238892, 2141549, 2044206, 1946863, 1849519], dtype=float)
    workbook_yield = np.array([15.0, 67.0, 78.0, 89.0, 101.0, 107.0], dtype=float)

    lock_fy24 = st.session_state.get("lock_fy24", True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{focus_lbl} revenue (AED)", f"{revenue[focus_idx]:,.0f}")
    c2.metric("Property value (AED)", f"{nbv[focus_idx]:,.0f}")
    c3.metric(f"Yield ({focus_lbl})", f"{workbook_yield[focus_idx]:.2f}%")
    c4.metric(f"Annual growth rate ({focus_lbl}→FY29)",
              f"{(cagr(revenue[focus_idx], revenue[-1], max(1,(len(YEARS)-1-focus_idx)))*100):.1f}%")

    st.markdown("### Revenue mix: lease vs turnover — exact (with costs & breakeven)")
    fig1 = go.Figure()
    lock_labels = hover_lock_labels(revenue, lock_fy24)
    fig1.add_trace(go.Bar(x=YEARS, y=lease_rental, name="Lease rental income",
                          customdata=np.array(lock_labels)[:,None],
                          hovertemplate="Year %{x}<br>Lease rent: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
    fig1.add_trace(go.Bar(x=YEARS, y=turnover_rev, name="Turnover revenue",
                          customdata=np.array(lock_labels)[:,None],
                          hovertemplate="Year %{x}<br>Turnover: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
    fig1.add_trace(go.Scatter(x=YEARS, y=revenue, name="Total revenue", mode="lines+markers",
                              customdata=np.array(lock_labels)[:,None],
                              hovertemplate="Year %{x}<br>Total revenue: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
    fig1.add_trace(go.Scatter(x=YEARS, y=expenses, name="Costs", mode="lines+markers",
                              customdata=np.array(lock_labels)[:,None],
                              hovertemplate="Year %{x}<br>Costs: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
    be = compute_breakeven_point(revenue, expenses, YEARS)
    if be:
        fig1.add_trace(go.Scatter(x=[be["x"]], y=[be["y"]], mode="markers+text", marker=dict(color="red", size=12),
                                  text=[f"BE {be['fy']}"], textposition="top center",
                                  hovertemplate="Breakeven at %{customdata[0]}<br>%{customdata[1]}: %{customdata[2]:.2f} of year<br>Revenue = Costs = %{y:,.0f} AED<extra></extra>",
                                  customdata=[[be["date"], be["fy"], be["t"]]], name="Breakeven", showlegend=True, line=dict(width=0)))
    fig1.update_layout(barmode="stack", height=380, yaxis_title="AED", margin=dict(t=60))
    add_fy24_shading(fig1, lock_fy24)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### Revenue each year + cumulative — exact")
    cum_rev = np.cumsum(revenue)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=YEARS, y=revenue, name="YoY revenue",
                          customdata=np.array(lock_labels)[:,None],
                          hovertemplate="Year %{x}<br>Revenue: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
    fig2.add_trace(go.Scatter(x=YEARS, y=cum_rev, name="Cumulative revenue", mode="lines+text",
                              text=[f"{int(v):,}" for v in cum_rev], textposition="top center"))
    _ymax2 = max(float(max(revenue)), float(max(cum_rev))) * 1.22
    fig2.update_layout(height=380, yaxis_title="AED", margin=dict(t=80))
    fig2.update_yaxes(range=[0, _ymax2])
    add_fy24_shading(fig2, lock_fy24)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Costs & commercial yield — exact")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=YEARS, y=expenses, name="Costs (AED)",
                          text=[f"{int(v):,}" for v in expenses], textposition="outside", texttemplate="%{text}",
                          customdata=np.array(lock_labels)[:,None],
                          hovertemplate="Year %{x}<br>Costs: %{y:,.0f} AED<br>%{customdata[0]}<extra></extra>"))
    fig3.add_trace(go.Scatter(x=YEARS, y=workbook_yield, name="Commercial yield % (workbook)", mode="lines+markers", yaxis="y2"))
    _etop2 = float(max(expenses)) * 1.25
    fig3.update_layout(height=340, yaxis_title="AED", margin=dict(t=90), yaxis2=dict(title="Yield %", overlaying='y', side='right', range=[0, 125]))
    fig3.update_layout(yaxis=dict(range=[0, _etop2]))
    add_fy24_shading(fig3, lock_fy24)
    st.plotly_chart(fig3, use_container_width=True)

