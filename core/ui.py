# core/ui.py
import streamlit as st

def inject_css() -> None:
    """No-op for now (we can add CSS later)."""
    return

def kpi(title: str, value: str, sub: str = "") -> None:
    """Simple KPI card using built-in Streamlit components."""
    st.markdown(f"**{title}**")
    st.markdown(f"### {value}")
    if sub:
        st.caption(sub)

