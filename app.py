import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ENSO Agricultural Risk Predictor",
    page_icon="🌦️",
    layout="wide"
)

BASE_DIR        = Path(__file__).resolve().parent
DATA_PATH       = BASE_DIR / "final_crop_dataset_complete.csv"
PREDICT_SCRIPT  = BASE_DIR / "python" / "predict.py"
HASKELL_DIR     = BASE_DIR / "haskell"
HASKELL_SRC     = HASKELL_DIR / "crop_recommend.hs"
HASKELL_BINARY  = HASKELL_DIR / "crop_recommend"


@st.cache_resource
def compile_haskell():
    if HASKELL_BINARY.exists():
        return True, "Pre-compiled binary found."

    if not HASKELL_SRC.exists():
        return False, f"Haskell source not found at {HASKELL_SRC}"

    result = subprocess.run(
        ["ghc", "-O2", "-o", str(HASKELL_BINARY), str(HASKELL_SRC.name)],
        capture_output=True,
        text=True,
        cwd=str(HASKELL_DIR)
    )

    if result.returncode == 0:
        return True, "Compiled successfully."
    else:
        return False, result.stderr.strip()


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

    state_col  = find_column(df, {"state", "state_name", "state name"})
    season_col = find_column(df, {"season", "season_name"})
    year_col   = find_column(df, {"year", "crop_year", "cropyear"})
    crop_col   = find_column(df, {"crop", "crop_name", "crop name"})

    for col, name in [(state_col, "State"), (season_col, "Season"), (year_col, "Year")]:
        if col is None:
            raise ValueError(f"{name} column not found in dataset.")

    states  = sorted(df[state_col].dropna().astype(str).unique().tolist())
    seasons = sorted(df[season_col].dropna().astype(str).unique().tolist())
    years   = sorted(
        pd.to_numeric(df[year_col], errors="coerce")
        .dropna().astype(int).unique().tolist()
    )
    crops = (
        sorted(df[crop_col].dropna().astype(str).unique().tolist())
        if crop_col else []
    )

    return states, seasons, years, crops, state_col, season_col


def validate_combo(df, state_col, season_col, state, season):
    sub = df[
        (df[state_col].astype(str) == state) &
        (df[season_col].astype(str) == season)
    ]
    return len(sub) > 0, len(sub)


def run_python_prediction(state, season, year, enso_mode, manual_phase=None):
    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--state", state,
        "--season", season,
        "--year", str(year),
        "--enso-mode", enso_mode
    ]

    if enso_mode == "manual" and manual_phase:
        cmd.extend(["--manual-phase", manual_phase])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR)
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Python prediction failed.")

    return json.loads(result.stdout)


def run_haskell_recommendation(state, season, crop, binary_ready):
    if binary_ready and HASKELL_BINARY.exists():
        cmd = [str(HASKELL_BINARY), state, season, crop]
    else:
        cmd = ["runhaskell", HASKELL_SRC.name, state, season, crop]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(HASKELL_DIR),
        timeout=120
    )

    if result.returncode != 0:
        err = result.stderr.strip() if result.stderr else "Haskell recommendation failed."
        out = result.stdout.strip() if result.stdout else ""
        raise RuntimeError(f"{err}\n{out}")

    return result.stdout.strip()


def main():
    st.title("🌦️ ENSO Agricultural Risk Predictor")
    st.caption("Python predicts rainfall & ENSO → Haskell recommends crops.")

    binary_ready, compile_msg = compile_haskell()
    if binary_ready:
        st.sidebar.success(f"✅ Haskell: {compile_msg}")
    else:
        st.sidebar.warning(f"⚠️ Haskell using runhaskell (slower): {compile_msg}")

    try:
        states, seasons, years, crops, state_col, season_col = load_metadata()
        df_raw = load_dataset()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    latest_hist_year = max(years)

    with st.sidebar:
        st.header("⚙️ Inputs")

        state  = st.selectbox("State", states)
        season = st.selectbox("Season", seasons)
        crop   = st.selectbox("Crop", crops) if crops else st.text_input("Crop")

        valid_combo, record_count = validate_combo(df_raw, state_col, season_col, state, season)
        if valid_combo:
            st.success(f"✅ {record_count} records for this combination")
        else:
            st.error("❌ No data for this State + Season combination. Haskell will not run.")

        enso_mode    = st.selectbox("ENSO Mode", ["historical", "live", "manual"])
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

    st.subheader("📊 Dataset Coverage")
    c1, c2, c3 = st.columns(3)
    c1.metric("States", len(states))
    c2.metric("Seasons", len(seasons))
    c3.metric("Historical Years", f"{min(years)} – {max(years)}")

    st.markdown("---")

    if run_btn:
        try:
            with st.spinner("⏳ Running Python climate prediction..."):
                py_result = run_python_prediction(
                    state=state,
                    season=season,
                    year=year,
                    enso_mode=enso_mode,
                    manual_phase=manual_phase
                )
            st.success("✅ Python prediction completed.")

            st.markdown("## 🌧️ Climate Prediction")
            p1, p2, p3 = st.columns(3)
            p1.metric("Predicted Rainfall (mm)", py_result["predicted_rainfall_mm"])
            p2.metric("ENSO Phase", py_result["enso_phase"])
            p3.metric("Rainfall Category", py_result["rainfall_category"])

            p4, p5, p6 = st.columns(3)
            p4.metric("Historical Normal (mm)", py_result["historical_normal_mm"])
            p5.metric("Anomaly (%)", py_result["anomaly_pct"])
            p6.metric("ONI Value", py_result["oni_value"] if py_result["oni_value"] is not None else "N/A")

            with st.expander("📄 Full JSON response"):
                st.json(py_result)

            st.markdown("---")

            with st.spinner("⏳ Running Haskell crop recommendation..."):
                hs_output = run_haskell_recommendation(state, season, crop, binary_ready)
            st.success("✅ Haskell recommendation completed.")

            st.markdown("## 🌾 Crop Recommendation")
            st.code(hs_output, language="text")

        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")

    else:
        st.markdown("## How it works")
        st.write(
            "1. Choose State, Season, Crop, and ENSO mode.\n"
            "2. Python predicts ENSO and rainfall and writes `forecast.json`.\n"
            "3. Haskell reads `forecast.json` and returns crop risk + recommendations.\n"
            "4. Results appear here."
        )


if __name__ == "__main__":
    main()
