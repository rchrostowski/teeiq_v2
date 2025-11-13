# core/ui.py
import streamlit as st


CSS = """
<style>
html, body, [class*="css"] {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.main {
  background-color: #f8f9fa;
}
h1, h2, h3 {
  letter-spacing: .02em;
}
.kpi-card {
  background: linear-gradient(135deg, #143d2a, #1f5c3e);
  border-radius: 18px;
  padding: 16px 18px;
  color: #f8f9fa;
  border: 1px solid #2f6b49;
}
.kpi-label {
  font-size: 0.85rem;
  text-transform: uppercase;
  opacity: 0.8;
}
.kpi-value {
  font-size: 1.6rem;
  font-weight: 600;
}
.kpi-sub {
  font-size: 0.9rem;
  opacity: 0.85;
}
.section-header {
  margin-top: 1.5rem;
  margin-bottom: 0.4rem;
  font-size: 1.1rem;
  font-weight: 600;
}
.badge-ai {
  display: inline-block;
  background: #e9f5ef;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.8rem;
  color: #1f5c3e;
  border: 1px solid #c6dfd0;
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
