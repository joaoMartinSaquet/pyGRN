import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="📊 Logfile Dashboard", layout="wide")

# Auto-refresh every 10 seconds
count = st_autorefresh(interval=10_000, key="log_autorefresh")

st.title("📊 Logfile Dashboard")

# --- Get logfile path from command-line arguments ---
if len(sys.argv) > 1:
    log_path = Path(sys.argv[1])
else:
    st.error("❌ No log file provided. Run with:\n\n`streamlit run app.py /path/to/logfile.log`")
    st.stop()

# --- Read and parse logfile ---
if log_path.exists():
    with open(log_path, "r") as f:
        lines = f.read().splitlines()

    # Separate S and G records
    s_records, g_records = [], []

    for line in lines:
        parts = line.split(",")
        if parts[0] == "S":
            s_records.append(parts)
        elif parts[0] == "G":
            g_records.append(parts)

    # Column definitions
    s_cols = [
        "type", "timestamp", "run_id", "index",
        "id", "species size", "adjusted fit", "best fit", 
        "best size", "repr distance", "min size of species"
    ]

    g_cols = [
        "type", "timestamp", "generation", "population_size",
        "best_fitness", "best_grn_size", "fit_mean", "fit_std"
    ]

    df_s = pd.DataFrame(s_records, columns=s_cols)
    df_g = pd.DataFrame(g_records, columns=g_cols)

    # Convert types
    for df in [df_s, df_g]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for col in df.columns[2:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Show species records
    st.subheader("🐠 Species Records (S)")
    st.dataframe(df_s.tail(20))

    # Show global records
    st.subheader("🌍 Global Records (G)")
    st.dataframe(df_g.tail(20))

    # Global metrics chart
    st.subheader("📈 Global Metrics Over Generations")
    fig, ax = plt.subplots()
    ax.plot(df_g["generation"], df_g["best_fitness"], label="Best Fitness")
    ax.plot(df_g["generation"], df_g["fit_mean"], label="Mean Fitness")
    ax.legend()
    st.pyplot(fig)

    # Species metric distribution
    st.subheader("📊 Species Value Distribution")
    metric_choice = st.selectbox(
        "Choose a metric", 
        ["adjusted fit", "best fit", "best size", "repr distance", "min size of species"]
    )
    fig2, ax2 = plt.subplots()
    ax2.hist(df_s[metric_choice].dropna(), bins=30)
    ax2.set_title(f"Distribution of {metric_choice}")
    st.pyplot(fig2)

else:
    st.error(f"❌ Log file not found: {log_path}")