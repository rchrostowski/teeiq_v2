# app.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from core.data import (
  clean_teetimes,
  kpis,
  daily_utilization,
  utilization_matrix_hour,
)
from core.pricing import compute_pricing_actions
from core.ui import inject_css, kpi


st.set_page_config(
  page_title="TeeIQ – Run your course like a hedge fund",
  page_icon="⛳",
  layout="wide",
)


def make_demo_data() -> pd.DataFrame:
  rng = pd.date_range("2024-07-01", periods=14 * 12 * 6, freq="10min")
  df = pd.DataFrame({"tee_time": rng})
  df = df[df["tee_time"].dt.hour.between(6, 18)]

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


def build_text_report(df: pd.DataFrame, slot_minutes: int) -> str:
  """Create a simple text report summarizing performance & pricing actions."""
  total, booked, util, revenue, potential = kpis(df)
  gap = potential - revenue

  today_date = df["date"].max()
  today_df = df[df["date"] == today_date]
  t_total, t_booked, t_util, t_rev, t_pot = kpis(today_df) if not today_df.empty else (0, 0, 0.0, 0.0, 0.0)
  t_gap = t_pot - t_rev

  trend = daily_utilization(df)
  avg_util_7d = trend["util"].tail(7).mean() * 100 if len(trend) >= 1 else 0.0

  # basic pricing actions (top 10 softest slots)
  actions = compute_pricing_actions(df, slot_minutes=slot_minutes, target_util=0.75, top_n=10)

  lines = []
  lines.append("TeeIQ Weekly Performance Report")
  lines.append("=" * 32)
  lines.append("")
  lines.append(f"Tee-time interval: {slot_minutes} minutes")
  lines.append("")
  lines.append("Overall performance")
  lines.append("-------------------")
  lines.append(f"Total slots:       {total:,}")
  lines.append(f"Booked slots:      {booked:,}")
  lines.append(f"Utilization:       {util*100:.1f}%")
  lines.append(f"Booked revenue:    ${revenue:,.0f}")
  lines.append(f"Potential revenue: ${potential:,.0f}")
  lines.append(f"Revenue gap:       ${gap:,.0f}")
  lines.append("")
  lines.append("Most recent day")
  lines.append("---------------")
  lines.append(f"Date:              {today_date}")
  lines.append(f"Slots:             {t_total:,}")
  lines.append(f"Booked slots:      {t_booked:,}")
  lines.append(f"Utilization:       {t_util*100:.1f}%")
  lines.append(f"Booked revenue:    ${t_rev:,.0f}")
  lines.append(f"Revenue gap:       ${t_gap:,.0f}")
  lines.append("")
  lines.append("Recent trend")
  lines.append("------------")
  lines.append(f"Average utilization over last 7 days: {avg_util_7d:.1f}%")
  lines.append("")
  lines.append("AI pricing suggestions (top softest blocks)")
  lines.append("--------------------------------------------")

  if actions.empty:
    lines.append("No pricing suggestions available (not enough data).")
  else:
    lines.append(f"Showing {len(actions)} softest blocks by expected utilization:")
    lines.append("")
    for _, row in actions.iterrows():
      lines.append(
        f"- {row['Weekday']} @ {row['Time']}: "
        f"Util {row['Expected Utilization']}, "
        f"Avg {row['Average Price']}, "
        f"Discount {row['Suggested Discount']}, "
        f"New {row['New Price']}"
      )

  lines.append("")
  lines.append("Next steps")
  lines.append("----------")
  lines.append("1) Apply recommended pricing on softest blocks.")
  lines.append("2) Monitor utilization and revenue over the next 7 days.")
  lines.append("3) Re-run TeeIQ to update your pricing plan.")

  return "\n".join(lines)


def main() -> None:
  inject_css()

  st.title("TeeIQ")
  st.caption("Run your course like a hedge fund. Slot-level pricing intelligence for golf tee sheets.")

  # --- Sidebar: data source ---
  with st.sidebar:
    st.subheader("Data")
    tee_file = st.file_uploader("Upload tee-sheet CSV", type=["csv"])
    use_demo = st.checkbox("Use demo data", value=not bool(tee_file))

  if tee_file is not None:
    raw_df = pd.read_csv(tee_file)
  elif use_demo:
    raw_df = make_demo_data()
  else:
    raw_df = None

  if raw_df is None:
    st.info("Upload a tee-sheet CSV or check 'Use demo data'.")
    return

  # --- Clean data ---
  try:
    df = clean_teetimes(raw_df)
  except Exception as e:
    st.error(f"Data error: {e}")
    return

  # --- Sidebar: tee time spacing ---
  slot_minutes = st.sidebar.selectbox(
    "Tee-time interval (minutes)",
    options=[5, 7, 8, 9, 10, 12, 15],
    index=[5, 7, 8, 9, 10, 12, 15].index(10),
    help="Match your tee-sheet spacing.",
  )

  # --- Top KPIs ---
  total, booked, util, revenue, potential = kpis(df)
  c1, c2, c3 = st.columns(3)
  with c1:
    kpi("Utilization", f"{util*100:.0f}%", f"{booked:,} of {total:,} slots booked")
  with c2:
    kpi("Booked Revenue", f"${revenue:,.0f}", f"Potential at rack: ${potential:,.0f}")
  with c3:
    gap = potential - revenue
    kpi("Revenue Gap", f"${gap:,.0f}", "Lost to empty or underpriced tee times")

  # --- Tabs ---
  tab_dash, tab_pricing, tab_util, tab_reports = st.tabs(
    ["Executive Dashboard", "Pricing AI", "Utilization", "Reports"]
  )

  # --- EXECUTIVE DASHBOARD ---
  with tab_dash:
    st.markdown('<div class="section-header">Booking Trend</div>', unsafe_allow_html=True)
    trend = daily_utilization(df)
    if trend.empty:
      st.info("Not enough data to show a trend.")
    else:
      fig, ax = plt.subplots(figsize=(8, 3))
      ax.plot(trend["date"], trend["util"] * 100, marker="o")
      ax.set_ylabel("Utilization (%)")
      ax.set_xlabel("Date")
      ax.grid(True, alpha=0.3)
      st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="section-header">Most Recent Day Snapshot</div>', unsafe_allow_html=True)
    today = df[df["date"] == df["date"].max()]
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
        df, slot_minutes=slot_minutes, target_util=target_util, top_n=top_n,
      )
      if actions.empty:
        st.info("No actions available (no data).")
      else:
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

  # --- UTILIZATION ---
  with tab_util:
    st.markdown('<div class="section-header">Weekly Utilization Heatmap (by hour)</div>', unsafe_allow_html=True)
    mat = utilization_matrix_hour(df)
    if mat.empty:
      st.info("No utilization data.")
    else:
      data = mat.to_numpy()
      fig2, ax2 = plt.subplots(figsize=(12, 3))
      im = ax2.imshow(data, aspect="auto")
      ax2.set_yticks(range(len(mat.index)))
      ax2.set_yticklabels(mat.index)

      hours = list(mat.columns)
      labels = []
      for h in hours:
        hh = h % 12 or 12
        ampm = "AM" if h < 12 else "PM"
        labels.append(f"{hh}{ampm}")
      ax2.set_xticks(range(len(hours)))
      ax2.set_xticklabels(labels)
      ax2.set_xlabel("Hour of day")
      cbar = fig2.colorbar(im, ax=ax2, fraction=0.02, pad=0.02)
      cbar.set_label("Utilization")

      for i in range(data.shape[0]):
        for j in range(data.shape[1]):
          v = data[i, j]
          if not np.isnan(v):
            ax2.text(
              j, i, f"{v*100:.0f}%",
              ha="center", va="center",
              fontsize=7, color="white",
            )

      st.pyplot(fig2, use_container_width=True)

  # --- REPORTS ---
  with tab_reports:
    st.markdown('<div class="section-header">Generate Weekly Report</div>', unsafe_allow_html=True)
    st.write(
      "This creates a simple text report summarizing utilization, revenue, and AI pricing suggestions. "
      "You can download it and share with your team."
    )

    if st.button("Generate report"):
      report_text = build_text_report(df, slot_minutes)
      st.text_area("Report preview", value=report_text, height=300)
      st.download_button(
        "Download report (.txt)",
        data=report_text.encode(),
        file_name="teeiq_report.txt",
        mime="text/plain",
      )


if __name__ == "__main__":
  main()



if __name__ == "__main__":
  main()

