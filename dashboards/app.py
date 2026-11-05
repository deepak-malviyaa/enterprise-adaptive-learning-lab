from pathlib import Path
import json
import streamlit as st
st.set_page_config(page_title="Adaptive Learning Lab", layout="wide")
st.title("Enterprise Adaptive Learning Laboratory")
st.caption("Stored experiment artifacts only; this dashboard does not run training.")
runs = sorted(Path("outputs/runs").glob("*/metrics.json"), reverse=True)
if not runs:
    st.info("No completed runs found. Execute the synthetic quickstart first.")
else:
    selected = st.selectbox("Run", runs, format_func=lambda p: p.parent.name)
    st.json(json.loads(selected.read_text(encoding="utf-8")))
