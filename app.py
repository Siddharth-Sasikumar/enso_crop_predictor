import json
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
HASKELL_DIR = BASE_DIR / "haskell"
HASKELL_BINARY = HASKELL_DIR / "crop_recommend"
DATASET_PATH = BASE_DIR / "final_crop_dataset_complete.csv"
FORECAST_PATH = BASE_DIR / "forecast.json"


st.set_page_config(page_title="ENSO Crop Predictor", layout="wide")
st.title("ENSO Crop Predictor")


@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)


def load_forecast():
    if FORECAST_PATH.exists():
        try:
            with open(FORECAST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def run_haskell_crop_recommend(state, season, crop):
    if not HASKELL_BINARY.exists():
        return {"error": f"Haskell binary not found: {HASKELL_BINARY}"}

    try:
        result = subprocess.run(
            [str(HASKELL_BINARY), state, season, crop],
            cwd=str(HASKELL_DIR),
            capture_output=True,
            text=True,
            timeout=60
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

        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON returned by Haskell program",
                "raw_output": output
            }

    except Exception as e:
        return {"error": str(e)}


df = load_data()
forecast = load_forecast()

st.subheader("Dataset Preview")
st.dataframe(df.head())

states = sorted(df["State"].dropna().unique().tolist()) if "State" in df.columns else []
seasons = sorted(df["Season"].dropna().unique().tolist()) if "Season" in df.columns else []
crops = sorted(df["Crop"].dropna().unique().tolist()) if "Crop" in df.columns else []

st.subheader("Crop Recommendation")

col1, col2, col3 = st.columns(3)

with col1:
    selected_state = st.selectbox("State", states if states else ["Tamil Nadu"])

with col2:
    selected_season = st.selectbox("Season", seasons if seasons else ["Kharif"])

with col3:
    selected_crop = st.selectbox("Crop", crops if crops else ["Rice"])

if st.button("Predict / Recommend"):
    response = run_haskell_crop_recommend(
        selected_state,
        selected_season,
        selected_crop
    )

    if "error" in response:
        st.error(response["error"])
        if "stderr" in response:
            st.code(response["stderr"])
        if "stdout" in response:
            st.code(response["stdout"])
        if "raw_output" in response:
            st.code(response["raw_output"])
    else:
        st.success("Prediction generated successfully")
        st.json(response)

        if isinstance(response, dict):
            if "crop" in response:
                st.write(f"**Crop:** {response['crop']}")
            if "state" in response:
                st.write(f"**State:** {response['state']}")
            if "season" in response:
                st.write(f"**Season:** {response['season']}")
            if "ensoPhase" in response:
                st.write(f"**ENSO Phase:** {response['ensoPhase']}")
            if "expectedYield" in response:
                st.write(f"**Expected Yield:** {response['expectedYield']}")
            if "riskLevel" in response:
                st.write(f"**Risk Level:** {response['riskLevel']}")
            if "explanation" in response:
                st.info(response["explanation"])

st.subheader("Forecast Data")
if forecast:
    st.json(forecast)
else:
    st.info("No forecast.json found or file is empty.")
