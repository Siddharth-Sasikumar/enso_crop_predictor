import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ENSO Agricultural Risk Predictor",
    page_icon="🌾",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "final_crop_dataset_complete.csv"
PREDICT_SCRIPT = BASE_DIR / "python" / "predict.py"
HASKELL_DIR = BASE_DIR / "haskell"
HASKELL_SCRIPT = HASKELL_DIR / "crop_recommend.hs"


st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #9ca3af;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}

.status-bar {
    padding: 14px 18px;
    border-radius: 14px;
    margin: 12px 0 18px 0;
    font-weight: 600;
    border: 1px solid transparent;
}

.status-good {
    background: linear-gradient(90deg, rgba(34,197,94,0.18), rgba(34,197,94,0.07));
    border-color: rgba(34,197,94,0.35);
    color: #dcfce7;
}

.status-bad {
    background: linear-gradient(90deg, rgba(239,68,68,0.18), rgba(239,68,68,0.07));
    border-color: rgba(239,68,68,0.35);
    color: #fee2e2;
}

.status-warn {
    background: linear-gradient(90deg, rgba(245,158,11,0.18), rgba(245,158,11,0.07));
    border-color: rgba(245,158,11,0.35);
    color: #fef3c7;
}

.summary-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.70), rgba(15,23,42,0.92));
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 18px;
    padding: 20px 22px;
    margin: 14px 0 18px 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

.summary-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 10px;
    color: #f8fafc;
}

.summary-pre {
    white-space: pre-wrap;
    font-family: "Courier New", monospace;
    font-size: 0.96rem;
    line-height: 1.7;
    margin: 0;
    color: #e5e7eb;
}

.section-card-good {
    background: linear-gradient(135deg, rgba(34,197,94,0.09), rgba(22,163,74,0.04));
    border: 1px solid rgba(34,197,94,0.22);
    border-radius: 18px;
    padding: 16px 16px 8px 16px;
    margin-top: 10px;
    margin-bottom: 18px;
}

.section-card-bad {
    background: linear-gradient(135deg, rgba(239,68,68,0.09), rgba(185,28,28,0.04));
    border: 1px solid rgba(239,68,68,0.22);
    border-radius: 18px;
    padding: 16px 16px 8px 16px;
    margin-top: 10px;
    margin-bottom: 18px;
}

.section-title-good {
    font-size: 1.2rem;
    font-weight: 800;
    color: #86efac;
    margin-bottom: 10px;
}

.section-title-bad {
    font-size: 1.2rem;
    font-weight: 800;
    color: #fca5a5;
    margin-bottom: 10px;
}

.metric-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.70), rgba(15,23,42,0.92));
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 16px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def find_column(df, candidates):
    for c in df.columns:
        if c.lower().strip() in candidates:
            return c
    return None


@st.cache_data
def load_metadata():
    df = load_dataset()

    state_col = find_column(df, {"state", "state_name", "state name"})
    season_col = find_column(df, {"season", "season_name"})
    year_col = find_column(df, {"year", "crop_year", "cropyear"})
    crop_col = find_column(df, {"crop", "crop_name", "crop name"})

    for col, name in [(state_col, "State"), (season_col, "Season"), (year_col, "Year")]:
        if col is None:
            raise ValueError(f"{name} column not found in dataset.")

    states = sorted(df[state_col].dropna().astype(str).unique().tolist())
    seasons = sorted(df[season_col].dropna().astype(str).unique().tolist())
    years = sorted(
        pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int).unique().tolist()
    )
    crops = sorted(df[crop_col].dropna().astype(str).unique().tolist()) if crop_col else []

    return states, seasons, years, crops, state_col, season_col


def validate_combo(df, state_col, season_col, state, season):
    sub = df[
        (df[state_col].astype(str) == state) &
        (df[season_col].astype(str) == season)
    ]
    return len(sub) > 0, len(sub)


def run_python_prediction(state, season, year, enso_mode, manual_phase=None):
    cmd = [
        "python", str(PREDICT_SCRIPT),
        "--state", state,
        "--season", season,
        "--year", str(year),
        "--enso-mode", enso_mode
    ]

    if enso_mode == "manual" and manual_phase:
        cmd.extend(["--manual-phase", manual_phase])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Python prediction failed.")

    return json.loads(result.stdout)


def run_haskell_crop_recommend(state, season, crop):
    if not HASKELL_SCRIPT.exists():
        return {"error": f"Haskell script not found: {HASKELL_SCRIPT}"}

    try:
        result = subprocess.run(
            ["runhaskell", str(HASKELL_SCRIPT)],
            cwd=str(HASKELL_DIR),
            input=f"{state}\n{season}\n{crop}\n",
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return {
                "error": "Haskell program failed",
                "stderr": result.stderr,
                "stdout": result.stdout
            }

        output = result.stdout.strip()
        if not output:
            return {"error": "Haskell program returned empty output"}

        return {"output": output}

    except Exception as e:
        return {"error": str(e)}


def extract_section_lines(text, start_label):
    lines = text.splitlines()
    start_idx = None

    for i, line in enumerate(lines):
        if start_label.lower() in line.lower():
            start_idx = i + 1
            break

    if start_idx is None:
        return []

    collected = []
    for line in lines[start_idx:]:
        s = line.strip()

        if not s:
            continue

        if s.startswith("=============================="):
            break

        if s.lower().startswith("top 3 ") and start_label.lower() not in s.lower():
            break

        collected.append(s)

    return collected


def parse_crop_table(lines):
    rows = []

    pattern = re.compile(
        r"^(.*?)\s+\|\s+Risk:\s+([0-9.]+)\s+\((.*?)\)\s+\|\s+Est\.\s+Yield:\s+([0-9.]+)\s+tons/ha$"
    )

    for line in lines:
        m = pattern.match(line)
        if m:
            rows.append({
                "Crop": m.group(1).strip(),
                "Risk Score": float(m.group(2)),
                "Risk Level": m.group(3).strip(),
                "Estimated Yield (tons/ha)": float(m.group(4))
            })

    return pd.DataFrame(rows)


def split_haskell_output(raw_text):
    safer_lines = extract_section_lines(raw_text, "Top 3 Safer Alternatives")
    risky_lines = extract_section_lines(raw_text, "Top 3 Riskiest Crops")

    safer_df = parse_crop_table(safer_lines)
    risky_df = parse_crop_table(risky_lines)

    cleaned_text = []
    skip_modes = [
        "top 3 safer alternatives",
        "top 3 riskiest crops"
    ]

    for line in raw_text.splitlines():
        low = line.lower().strip()

        if any(low.startswith(x) for x in skip_modes):
            continue

        if "| Risk:" in line and "| Est. Yield:" in line:
            continue

        cleaned_text.append(line)

    return {
        "safer_df": safer_df,
        "risky_df": risky_df,
        "cleaned_text": "\n".join(cleaned_text).strip()
    }


def make_status_badge(text):
    t = text.lower()
    if "not cultivated" in t or "not suitable" in t:
        return "bad", "🔴 High caution"
    if "safe" in t or "safer" in t or "suitable" in t:
        return "good", "🟢 Favourable"
    return "warn", "🟡 Mixed / review needed"


def styled_dataframe(df, good=True):
    if df.empty:
        return df

    df = df.copy().reset_index(drop=True)
    df.index = df.index + 1
    df["Risk Score"] = df["Risk Score"].map(lambda x: f"{x:.2f}")
    df["Estimated Yield (tons/ha)"] = df["Estimated Yield (tons/ha)"].map(lambda x: f"{x:.2f}")

    def color_risk_level(val):
        v = str(val).upper()
        if v in ["LOW", "VERY LOW"]:
            return "color: #86efac; font-weight: 700;"
        if v in ["MEDIUM", "MODERATE"]:
            return "color: #fcd34d; font-weight: 700;"
        if v in ["HIGH", "VERY HIGH"]:
            return "color: #fca5a5; font-weight: 700;"
        return ""

    def color_crop(_):
        return "color: #bbf7d0; font-weight: 600;" if good else "color: #fecaca; font-weight: 600;"

    return df.style \
        .applymap(color_crop, subset=["Crop"]) \
        .applymap(color_risk_level, subset=["Risk Level"])


def main():
    st.markdown('<div class="main-title">🌾 Crop Recommendation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">ENSO-based agricultural risk and crop suitability dashboard</div>', unsafe_allow_html=True)

    try:
        states, seasons, years, crops, state_col, season_col = load_metadata()
        df_raw = load_dataset()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    latest_hist_year = max(years)

    with st.sidebar:
        st.header("⚙️ Inputs")

        state = st.selectbox("State", states)
        season = st.selectbox("Season", seasons)
        crop = st.selectbox("Crop", crops) if crops else st.text_input("Crop")

        valid_combo, record_count = validate_combo(df_raw, state_col, season_col, state, season)
        if valid_combo:
            st.success(f"✅ {record_count} records found")
        else:
            st.error("❌ No data for this State + Season combination")

        enso_mode = st.selectbox("ENSO Mode", ["historical", "live", "manual"])
        manual_phase = None

        if enso_mode == "historical":
            year = st.selectbox("Historical Year", years, index=len(years) - 1)
        elif enso_mode == "live":
            year = st.number_input(
                "Prediction Year",
                min_value=latest_hist_year + 1,
                max_value=2100,
                value=max(2026, latest_hist_year + 1),
                step=1
            )
        else:
            year = st.number_input(
                "Scenario Year",
                min_value=latest_hist_year + 1,
                max_value=2100,
                value=max(2026, latest_hist_year + 1),
                step=1
            )
            manual_phase = st.selectbox("Manual ENSO Phase", ["El Nino", "Neutral", "La Nina"])

        run_btn = st.button(
            "▶ Run Full Prediction",
            type="primary",
            use_container_width=True,
            disabled=not valid_combo
        )

    st.subheader("📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("States", len(states))
    c2.metric("Seasons", len(seasons))
    c3.metric("Historical Years", f"{min(years)} – {max(years)}")

    st.markdown("---")

    if run_btn:
        try:
            with st.spinner("Running Python climate prediction..."):
                py_result = run_python_prediction(
                    state=state,
                    season=season,
                    year=year,
                    enso_mode=enso_mode,
                    manual_phase=manual_phase
                )

            st.subheader("🌧️ Climate Prediction")
            p1, p2, p3 = st.columns(3)
            p1.metric("Predicted Rainfall (mm)", py_result["predicted_rainfall_mm"])
            p2.metric("ENSO Phase", py_result["enso_phase"])
            p3.metric("Rainfall Category", py_result["rainfall_category"])

            p4, p5, p6 = st.columns(3)
            p4.metric("Historical Normal (mm)", py_result["historical_normal_mm"])
            p5.metric("Anomaly (%)", py_result["anomaly_pct"])
            p6.metric("ONI Value", py_result["oni_value"] if py_result["oni_value"] is not None else "N/A")

            st.markdown("---")

            with st.spinner("Running Haskell crop recommendation..."):
                hs_result = run_haskell_crop_recommend(state, season, crop)

            if "error" in hs_result:
                st.error(f"❌ Haskell failed: {hs_result['error']}")
                if "stderr" in hs_result and hs_result["stderr"]:
                    st.code(hs_result["stderr"], language="text")
                if "stdout" in hs_result and hs_result["stdout"]:
                    st.code(hs_result["stdout"], language="text")
            else:
                parsed = split_haskell_output(hs_result["output"])

                summary_lines = []
                for line in parsed["cleaned_text"].splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if "| Risk:" in line and "| Est. Yield:" in line:
                        continue
                    if line.startswith("==="):
                        continue
                    summary_lines.append(line)

                summary_text = "\n".join(summary_lines)

                badge_type, badge_text = make_status_badge(summary_text)
                badge_class = {
                    "good": "status-good",
                    "bad": "status-bad",
                    "warn": "status-warn"
                }[badge_type]

                st.markdown(
                    f'<div class="status-bar {badge_class}">{badge_text}</div>',
                    unsafe_allow_html=True
                )

                st.markdown("## 🌾 Crop Recommendation")

                if summary_text:
                    st.markdown(
                        f"""
<div class="summary-card">
    <div class="summary-title">Recommendation Summary</div>
    <pre class="summary-pre">{summary_text}</pre>
</div>
""",
                        unsafe_allow_html=True
                    )

                st.markdown('<div class="section-card-good">', unsafe_allow_html=True)
                st.markdown('<div class="section-title-good">✅ Recommended Alternatives</div>', unsafe_allow_html=True)
                if not parsed["safer_df"].empty:
                    st.dataframe(
                        styled_dataframe(parsed["safer_df"], good=True),
                        use_container_width=True,
                        height=180
                    )
                else:
                    st.info("No recommended alternatives found.")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-card-bad">', unsafe_allow_html=True)
                st.markdown('<div class="section-title-bad">⚠️ High-Risk Crops</div>', unsafe_allow_html=True)
                if not parsed["risky_df"].empty:
                    st.dataframe(
                        styled_dataframe(parsed["risky_df"], good=False),
                        use_container_width=True,
                        height=180
                    )
                else:
                    st.info("No high-risk crops found.")
                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("View raw Haskell output"):
                    st.code(hs_result["output"], language="text")

        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")

    else:
        st.info(
            "Select State, Season, Crop, and ENSO mode from the sidebar, then click 'Run Full Prediction'."
        )


if __name__ == "__main__":
    main()
