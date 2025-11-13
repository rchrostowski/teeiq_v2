# app.py
import streamlit as st
import matplotlib.pyplot as plt

from core.data import clean_teetimes, kpis, daily_utilization, utilization_matrix_hour
from core.pricing import compute_pricing_actions
from core.ui import inject_css, kpi


st.set_page_config(
    page_title="TeeIQ – Run your course like a hedge fund",
    page_icon="⛳",
    layout="wide",
)

inject_css()

st.title("TeeIQ")
st.caption("Run your course like a hedge fund. Slot-level pricing intelligence for golf tee sheets.")


# ---------- Data upload / demo ----------
with st.sidebar:
    st.subheader("Data")
    tee_file = st.file_uploader("Upload tee-sheet CSV", type=["csv"])
    use_demo = st.checkbox("Use demo data", value=not bool(tee_file))

def make_demo_data():
    import pandas as pd
    import numpy as np
    rng = pd.date_range("2024-07-01", periods=14, freq="10min")
    df = pd.DataFrame({"tee_time": rng})
    df = df[df["tee_time"].dt.hour.between(6, 18)]
    # fake booked pattern
    df["booked"] = np.where(
        (df["tee_time"].dt.hour.between(7, 10))
        | (df["tee_time"].dt.hour.between(14, 16)),
        np.random.binomial(1, 0.8, size=len(df)),
        np.random.binomial(1, 0.35, size=len(df)),
    )
    base_price = np.where(df["tee_time"].dt.hour < 12, 90, 75)
    noise = np.random.normal(0, 8, size=len(df))
    df["price"] = base_price + noise
    return df

if tee_file is not None:
    import pandas as pd
    raw_df = pd.read_csv(tee_file)
elif use_demo:
    raw_df = make_demo_data()
else:
    raw_df = None

if raw_df is None:
    st.info("Upload a tee-sheet CSV in the sidebar or check 'Use demo data'.")
    st.stop()

# clean + standardize
try:
    df = clean_teetimes(raw_df)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

slot_minutes = st.sidebar.selectbox(
    "Tee-time interval",
    options=[5, 7, 8, 9, 10, 12, 15],
    index=[5, 7, 8, 9, 10, 12, 15].index(10),
    help="Match your tee-sheet spacing.",
)

# ---------- KPIs row ----------
total, booked, util, revenue, potential = kpis(df)
c1, c2, c3 = st.columns(3)
with c1:
    kpi("Utilization", f"{util*100:.0f}%", f"{booked:,} of {total:,} slots booked")
with c2:
    kpi("Booked Revenue", f"${revenue:,.0f}", f"Potential at rack: ${potential:,.0f}")
with c3:
    gap = potential - revenue
    kpi("Revenue Gap", f"${gap:,.0f}", "Lost to empty or underpriced tee times")


# ---------- Tabs ----------
tab_dash, tab_pricing, tab_util, tab_reports = st.tabs(
    ["Executive Dashboard", "Pricing AI", "Utilization", "Reports"]
)

# --- EXECUTIVE DASHBOARD ---
with tab_dash:
    st.markdown('<div class="section-header">Booking Trend</div>', unsafe_allow_html=True)
    trend = daily_utilization(df)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(trend["date"], trend["util"] * 100, marker="o")
    ax.set_ylabel("Utilization (%)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="section-header">Today’s Performance Snapshot</div>', unsafe_allow_html=True)
    today = df[df["date"] == df["date"].max()]  # last day in data as "today"
    if today.empty:
        st.info("No data for the most recent day.")
    else:
        t_total, t_booked, t_util, t_rev, t_pot = kpis(today)
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            kpi("Today Utilization", f"{t_util*100:.0f}%", f"{t_booked:,}/{t_total:,} slots")
        with cc2:
            kpi("Today Revenue", f"${t_rev:,.0f}", "")
        with cc3:
            kpi("Today Gap", f"${(t_pot - t_rev):,.0f}", "")


# --- PRICING AI ---
with tab_pricing:
    st.markdown('<span class="badge-ai">AI HIO · Hole-in-One Insights</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Dynamic Pricing Suggestions</div>', unsafe_allow_html=True)

    colL, colR = st.columns([2, 1])
    with colL:
        top_n = st.slider("Number of softest blocks to show", 5, 30, 10, 1)
    with colR:
        target_util = st.slider("Target utilization", 0.5, 0.95, 0.75, 0.01)

    if st.button("Generate pricing actions"):
        actions = compute_pricing_actions(
            df, slot_minutes=slot_minutes, target_util=target_util, top_n=top_n
        )
        if actions.empty:
            st.info("No actions available (no data).")
        else:
            # Top single rec
            top_row = actions.iloc[0]
            st.subheader("Top Single Recommendation")
            st.markdown(
                f"**Block:** {top_row['Weekday']} @ {top_row['Time']}  \n"
                f"**Expected Utilization:** {top_row['Expected Utilization']}  \n"
                f"**Average Price:** {top_row['Average Price']}  \n"
                f"**Suggested Discount:** {top_row['Suggested Discount']}  \n"
                f"**New Price:** {top_row['New Price']}"
            )
            st.caption("These are your softest blocks. Discount and market them first.")

            st.dataframe(actions, use_container_width=True)
            st.download_button(
                "Download pricing plan (CSV)",
                data=actions.to_csv(index=False).encode(),
                file_name=f"teeiq_pricing_plan_{slot_minutes}min.csv",
                mime="text/csv",
            )

            # Simple bar chart
            chart_df = actions.copy()
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.bar(chart_df["Time"], chart_df["Expected Utilization"].str.rstrip('%').astype(float))
            ax2.set_ylabel("Expected Utilization (%)")
            ax2.set_xlabel("Time of Day (softest → hardest)")
            ax2.tick_params(axis="x", labelrotation=90)
            st.pyplot(fig2, use_container_width=True)


# --- UTILIZATION ---
with tab_util:
    st.markdown('<div class="section-header">Weekly Utilization Heatmap (by hour)</div>', unsafe_allow_html=True)
    mat = utilization_matrix_hour(df)
    if mat.empty:
        st.info("No utilization data.")
    else:
        import numpy as np
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(mat.to_numpy(), aspect="auto")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        hours = list(mat.columns)
        labels = []
        for h in hours:
            hh = h % 12 or 12
            ampm = "AM" if h < 12 else "PM"
            labels.append(f"{hh}{ampm}")
        ax.set_xticks(range(len(hours)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Hour of day")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Utilization")
        # annotate
        data = mat.to_numpy()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=7, color="white")
        st.pyplot(fig, use_container_width=True)


# --- REPORTS (stub for now) ---
with tab_reports:
    st.markdown('<div class="section-header">Reporting</div>', unsafe_allow_html=True)
    st.write("Coming next: weekly PDF with KPIs, trend charts, and pricing plan summary.")
    st.write("For now, download your pricing plan from the Pricing AI tab and share with your team.")
