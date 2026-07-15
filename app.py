"""
Rainfall Prediction App (Purple/Blue Glitter, No-Sidebar / App-Style Navigation)

What this file does:
- Loads the model that train.py already trained and saved
- Draws a purple/blue "glitter" themed website using Streamlit
- Instead of a sidebar, navigation works like a shopping app:
  you tap a tile on the Home screen and it takes you to a full page,
  and inside "Predict" the flow moves forward step by step
  (fill form -> tap Predict -> land on a Result page), similar to
  tapping "Pay" in a shopping app and landing on a confirmation page.
- Every page shows a top icon nav bar so you can jump anywhere.
"""

import json
import random

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# This must be the very first Streamlit command in the file
st.set_page_config(
    page_title="Rainfall Predictor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Purple / Blue "glitter" theme
# We build the glitter as hundreds of tiny CSS box-shadow dots (a common
# trick for a starfield/glitter effect) so there is no image to load.
# The seed keeps it stable between reruns instead of re-randomizing.
# ---------------------------------------------------------------------------


def _generate_glitter_dots(count, seed):
    rng = random.Random(seed)
    sparkle_colors = ["#ffffff", "#d6b8ff", "#9fd8ff", "#b98bff", "#7ec8ff", "#e0c3ff"]
    dots = []
    for _ in range(count):
        x = rng.randint(0, 100)
        y = rng.randint(0, 100)
        color = rng.choice(sparkle_colors)
        dots.append(f"{x}vw {y}vh {color}")
    return ", ".join(dots)


GLITTER_LAYER_SLOW = _generate_glitter_dots(90, seed=11)
GLITTER_LAYER_FAST = _generate_glitter_dots(55, seed=22)

st.markdown(
    f"""
<style>
/* ---------------- App background: deep purple -> blue gradient ---------------- */
.stApp {{
    background: linear-gradient(160deg, #1a0b2e 0%, #2d1257 25%, #3a1c71 45%,
                                 #1e3c72 70%, #142850 100%);
    background-attachment: fixed;
}}

/* ---------------- Glitter overlay (two twinkling star layers) ---------------- */
.stApp::before, .stApp::after {{
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 2px; height: 2px;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
}}
.stApp::before {{
    box-shadow: {GLITTER_LAYER_SLOW};
    animation: twinkle-slow 4s ease-in-out infinite alternate;
}}
.stApp::after {{
    box-shadow: {GLITTER_LAYER_FAST};
    animation: twinkle-fast 2.2s ease-in-out infinite alternate;
}}
@keyframes twinkle-slow {{
    from {{ opacity: 0.25; }}
    to   {{ opacity: 0.9; }}
}}
@keyframes twinkle-fast {{
    from {{ opacity: 0.85; }}
    to   {{ opacity: 0.35; }}
}}

/* Hide the default Streamlit chrome we don't want (sidebar arrow, menu, footer) */
section[data-testid="stSidebar"] {{ display: none !important; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ---------------- Top app bar ---------------- */
.app-topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 26px;
    margin: -1rem -1rem 18px -1rem;
    background: linear-gradient(90deg, rgba(58,28,113,0.85), rgba(20,40,80,0.85));
    border-bottom: 1px solid rgba(200,170,255,0.25);
    backdrop-filter: blur(6px);
    position: relative;
    z-index: 2;
}}
.app-topbar .brand {{
    font-size: 1.35rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e6c9ff, #9fd8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.5px;
}}
.app-topbar .tagline {{
    font-size: 0.8rem;
    color: #cbb8ff;
    opacity: 0.85;
}}

/* ---------------- Generic "glass card" ---------------- */
.glass-card {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(200,170,255,0.25);
    border-radius: 18px;
    padding: 22px;
    backdrop-filter: blur(6px);
    box-shadow: 0 4px 30px rgba(90, 30, 160, 0.25);
    position: relative;
    z-index: 1;
}}

/* ---------------- Text color fixes on dark background ---------------- */
h1, h2, h3, h4, h5, p, span, label, .stMarkdown, .stCaption {{
    color: #f1ecff;
}}
.stCaption, [data-testid="stCaptionContainer"] {{ color: #cbb8ff !important; }}

/* ---------------- Buttons: glowing purple/blue pill ---------------- */
.stButton > button {{
    width: 100%;
    border-radius: 14px;
    font-weight: 700;
    border: 1px solid rgba(200,170,255,0.4);
    background: linear-gradient(135deg, #7b2ff7, #2196f3);
    color: white;
    padding: 0.6em 1em;
    box-shadow: 0 0 14px rgba(123, 47, 247, 0.45);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 0 22px rgba(159, 216, 255, 0.75);
    border-color: #e6c9ff;
}}
.stButton > button:active {{ transform: translateY(0px) scale(0.98); }}

/* Secondary "back" style buttons */
.back-btn .stButton > button {{
    background: rgba(255,255,255,0.08);
    box-shadow: none;
    width: auto;
    padding: 0.35em 0.9em;
    font-weight: 600;
}}

/* Nav bar buttons: icon + label, pill-shaped */
.nav-icons .stButton > button {{
    font-size: 0.95rem;
    padding: 0.5em 0.3em;
    border-radius: 12px;
    background: rgba(255,255,255,0.05);
    box-shadow: none;
}}
.nav-icons .stButton > button:hover {{
    background: rgba(255,255,255,0.14);
    box-shadow: 0 0 14px rgba(159, 216, 255, 0.5);
}}

/* Inputs */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
.stSelectbox div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.07) !important;
    color: #f1ecff !important;
    border-radius: 10px !important;
    border: 1px solid rgba(200,170,255,0.3) !important;
}}

/* Metrics / expanders */
div[data-testid="stMetric"], div[data-testid="stExpander"] {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(200,170,255,0.25);
    border-radius: 14px;
    padding: 6px 10px;
}}

/* Home page hero image styling */
[data-testid="stImage"] img {{
    border-radius: 22px !important;
    border: 1px solid rgba(200,170,255,0.3);
    box-shadow: 0 8px 40px rgba(123, 47, 247, 0.35);
}}
.home-tile {{
    border-radius: 20px;
    padding: 26px 20px;
    text-align: center;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(200,170,255,0.3);
    box-shadow: 0 0 18px rgba(123, 47, 247, 0.2);
    height: 100%;
}}
.home-tile .icon {{ font-size: 2.4rem; }}
.home-tile .title {{ font-weight: 800; font-size: 1.05rem; margin-top: 6px; }}
.home-tile .desc {{ font-size: 0.85rem; color: #cbb8ff; margin-top: 4px; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load the saved model and helper files (only happens once, thanks to caching)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_the_trained_model():
    return joblib.load("models/best_pipeline.joblib")


@st.cache_data
def load_a_json_file(filepath):
    with open(filepath) as f:
        return json.load(f)


@st.cache_data
def load_sample_data():
    return pd.read_csv("models/sample_for_viz.csv")


trained_pipeline = load_the_trained_model()
metrics = load_a_json_file("models/metrics.json")
feature_importance = load_a_json_file("models/feature_importance.json")
city_list = load_a_json_file("models/locations.json")
wind_direction_list = load_a_json_file("models/wind_dirs.json")
sample_data = load_sample_data()

NUMBER_COLUMNS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
    "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm", "Month",
]
TEXT_COLUMNS = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def degrees_to_compass_direction(degrees):
    """Turns a wind direction number (0-360) into a label like 'NW'."""
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / 22.5) % 16
    return directions[index]


def get_live_weather(city_name):
    """Looks up a city, then fetches its current weather. Returns None if not found."""

    geocode_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city_name, "count": 1},
        timeout=8,
    ).json()

    if "results" not in geocode_response or len(geocode_response["results"]) == 0:
        return None, None

    place = geocode_response["results"][0]
    latitude = place["latitude"]
    longitude = place["longitude"]

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,pressure_msl,"
                       "wind_speed_10m,wind_direction_10m,cloud_cover,precipitation",
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
        },
        timeout=8,
    ).json()

    return place, weather_response


def make_prediction(inputs_dict):
    """Takes a dictionary of weather values and returns (yes_or_no, probability)."""
    one_row_table = pd.DataFrame([inputs_dict], columns=NUMBER_COLUMNS + TEXT_COLUMNS)
    probability_of_rain = trained_pipeline.predict_proba(one_row_table)[0, 1]
    yes_or_no = trained_pipeline.predict(one_row_table)[0]
    return yes_or_no, probability_of_rain


# ---------------------------------------------------------------------------
# App-style navigation (no sidebar) — session state drives which "page" shows
# ---------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"
if "autofill" not in st.session_state:
    st.session_state.autofill = {}
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()


def render_topbar(show_nav=True):
    """
    Brand header shown on every page.
    The icon+label nav row is shown everywhere EXCEPT the home page
    (home page only shows the big title + Predict Now button).
    """
    st.markdown(
        """
        <div class="app-topbar">
            <div>
                <div class="brand">🌧️ Rainfall Predictor</div>
                <div class="tagline">Next-day rain forecasts, powered by ML</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not show_nav:
        return

    st.markdown('<div class="nav-icons">', unsafe_allow_html=True)
    nav_cols = st.columns(5)
    icons = ["🏠", "🔮", "📊", "🧭", "ℹ️"]
    nav_labels = ["Home", "Predict", "Insights", "Explorer", "About"]
    targets = ["home", "predict_form", "insights", "explorer", "about"]
    current = st.session_state.page

    for col, icon, nav_label, target in zip(nav_cols, icons, nav_labels, targets):
        with col:
            is_active = (current == target) or (
                target == "predict_form" and current == "predict_result"
            )
            label = f"{icon} **{nav_label}**" if is_active else f"{icon} {nav_label}"
            if st.button(label, key=f"nav_{target}", use_container_width=True):
                go_to(target)
    st.markdown("</div>", unsafe_allow_html=True)


def render_back_button(target="home", label="← Back"):
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button(label, key=f"back_to_{target}"):
        go_to(target)
    st.markdown("</div>", unsafe_allow_html=True)


render_topbar(show_nav=(st.session_state.page != "home"))

# ---------------------------------------------------------------------------
# PAGE: Home — simple title + a single "Predict Now" call-to-action
# ---------------------------------------------------------------------------

if st.session_state.page == "home":
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<div style="padding-top:40px;">', unsafe_allow_html=True)
        st.image(
            "https://images.unsplash.com/photo-1519692933481-e162a57d6721?"
            "auto=format&fit=crop&w=900&q=80",
            use_column_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown(
            """
            <div style="padding: 80px 0 0 0;">
                <div style="display:inline-block; padding:6px 18px; border-radius:999px;
                            background:rgba(255,255,255,0.08); border:1px solid rgba(200,170,255,0.3);
                            font-size:0.8rem; color:#cbb8ff; letter-spacing:0.5px; margin-bottom:22px;">
                    ⚡ Powered by Machine Learning
                </div>
                <div style="font-size:3rem;">🌧️</div>
                <div style="font-size:2.8rem; font-weight:800; line-height:1.2; margin-top:10px;
                            background: linear-gradient(90deg, #e6c9ff, #9fd8ff);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;">
                    Rainfall Prediction
                </div>
                <div style="font-size:1.1rem; color:#cbb8ff; margin-top:14px; max-width:460px;">
                    Get tomorrow's rain forecast in seconds — just enter today's weather.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("🔮 Predict Now", key="home_predict_now", use_container_width=True):
            go_to("predict_form")

# ---------------------------------------------------------------------------
# PAGE: Predict — form step, then a separate Result page (like a checkout flow)
# ---------------------------------------------------------------------------

elif st.session_state.page == "predict_form":
    render_back_button("home")
    st.title("Will it rain tomorrow?")
    st.write(
        "Enter today's weather observations, or auto-fill from live conditions "
        "for a real city, then tap Predict to see your forecast."
    )

    with st.expander("⚡ Auto-fill from live weather (optional)", expanded=False):
        left_column, right_column = st.columns([3, 1])
        city_typed = left_column.text_input("City name", placeholder="e.g. Sydney, Melbourne, Perth")
        fetch_clicked = right_column.button("Fetch live data", use_container_width=True)

        if fetch_clicked and city_typed:
            with st.spinner("Fetching live conditions..."):
                place, weather = get_live_weather(city_typed)

            if place is None:
                st.warning("Couldn't find that city. Try a different spelling.")
            else:
                current = weather["current"]
                daily = weather["daily"]

                st.session_state.autofill = {
                    "MinTemp": daily["temperature_2m_min"][0],
                    "MaxTemp": daily["temperature_2m_max"][0],
                    "Humidity9am": current["relative_humidity_2m"],
                    "Humidity3pm": current["relative_humidity_2m"],
                    "Pressure9am": current["pressure_msl"],
                    "Pressure3pm": current["pressure_msl"],
                    "WindGustSpeed": current["wind_speed_10m"] * 1.3,
                    "WindSpeed9am": current["wind_speed_10m"],
                    "WindSpeed3pm": current["wind_speed_10m"],
                    "WindGustDir": degrees_to_compass_direction(current["wind_direction_10m"]),
                    "Cloud9am": round(current["cloud_cover"] / 12.5),
                    "Cloud3pm": round(current["cloud_cover"] / 12.5),
                    "Rainfall": current.get("precipitation", 0.0),
                    "Temp9am": current["temperature_2m"],
                    "Temp3pm": current["temperature_2m"],
                }

                st.success(
                    f"Loaded live conditions for {place['name']}, {place.get('country', '')}. "
                    "Sunshine/Evaporation aren't available live, so adjust those manually if you like."
                )

    auto = st.session_state.autofill

    column_1, column_2, column_3 = st.columns(3)

    with column_1:
        st.subheader("Temperature & Rain")
        min_temp = st.number_input("Min Temp (°C)", -10.0, 50.0, float(auto.get("MinTemp", 15.0)))
        max_temp = st.number_input("Max Temp (°C)", -10.0, 55.0, float(auto.get("MaxTemp", 25.0)))
        rainfall_today = st.number_input("Rainfall today (mm)", 0.0, 400.0, float(auto.get("Rainfall", 0.0)))
        rain_today_choice = st.selectbox("Did it rain today?", ["No", "Yes"])
        evaporation = st.number_input("Evaporation (mm)", 0.0, 50.0, 5.0)
        sunshine = st.slider("Sunshine (hours)", 0.0, 14.0, 7.0)

    with column_2:
        st.subheader("Humidity, Pressure & Cloud")
        humidity_9am = st.slider("Humidity 9am (%)", 0, 100, int(auto.get("Humidity9am", 70)))
        humidity_3pm = st.slider("Humidity 3pm (%)", 0, 100, int(auto.get("Humidity3pm", 50)))
        pressure_9am = st.number_input("Pressure 9am (hPa)", 950.0, 1050.0, float(auto.get("Pressure9am", 1015.0)))
        pressure_3pm = st.number_input("Pressure 3pm (hPa)", 950.0, 1050.0, float(auto.get("Pressure3pm", 1013.0)))
        cloud_9am = st.slider("Cloud cover 9am (oktas)", 0, 8, int(auto.get("Cloud9am", 4)))
        cloud_3pm = st.slider("Cloud cover 3pm (oktas)", 0, 8, int(auto.get("Cloud3pm", 4)))

    with column_3:
        st.subheader("Wind & Location")

        default_city_index = 0
        if "Sydney" in city_list:
            default_city_index = city_list.index("Sydney")
        location = st.selectbox("Location (station)", city_list, index=default_city_index)

        month = st.select_slider("Month", options=list(range(1, 13)), value=7)

        default_wind_index = 0
        auto_wind_direction = auto.get("WindGustDir")
        if auto_wind_direction in wind_direction_list:
            default_wind_index = wind_direction_list.index(auto_wind_direction)

        wind_gust_dir = st.selectbox("Wind gust direction", wind_direction_list, index=default_wind_index)
        wind_dir_9am = st.selectbox("Wind direction 9am", wind_direction_list, index=default_wind_index)
        wind_dir_3pm = st.selectbox("Wind direction 3pm", wind_direction_list, index=default_wind_index)
        wind_gust_speed = st.number_input("Wind gust speed (km/h)", 0.0, 150.0, float(auto.get("WindGustSpeed", 40.0)))
        wind_speed_9am = st.number_input("Wind speed 9am (km/h)", 0.0, 100.0, float(auto.get("WindSpeed9am", 15.0)))
        wind_speed_3pm = st.number_input("Wind speed 3pm (km/h)", 0.0, 100.0, float(auto.get("WindSpeed3pm", 20.0)))

    st.markdown("---")

    predict_clicked = st.button("🔮 Predict Rain Tomorrow", type="primary", use_container_width=True)

    if predict_clicked:
        inputs = {
            "MinTemp": min_temp, "MaxTemp": max_temp, "Rainfall": rainfall_today,
            "Evaporation": evaporation, "Sunshine": sunshine,
            "WindGustSpeed": wind_gust_speed, "WindSpeed9am": wind_speed_9am,
            "WindSpeed3pm": wind_speed_3pm, "Humidity9am": humidity_9am,
            "Humidity3pm": humidity_3pm, "Pressure9am": pressure_9am,
            "Pressure3pm": pressure_3pm, "Cloud9am": cloud_9am, "Cloud3pm": cloud_3pm,
            "Temp9am": auto.get("Temp9am", min_temp), "Temp3pm": auto.get("Temp3pm", max_temp),
            "Month": month, "Location": location, "WindGustDir": wind_gust_dir,
            "WindDir9am": wind_dir_9am, "WindDir3pm": wind_dir_3pm,
            "RainToday": 1 if rain_today_choice == "Yes" else 0,
        }

        prediction, probability = make_prediction(inputs)
        # Just like tapping "Pay" and landing on a confirmation page,
        # tapping Predict takes you to a dedicated Result page.
        st.session_state.last_result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "location": location,
        }
        go_to("predict_result")

# ---------------------------------------------------------------------------
# PAGE: Predict Result — the "confirmation" style page
# ---------------------------------------------------------------------------

elif st.session_state.page == "predict_result":
    render_back_button("predict_form", "← Edit inputs")

    result = st.session_state.last_result
    if result is None:
        st.info("No prediction yet — head back and fill in the form.")
    else:
        probability = result["probability"]
        prediction = result["prediction"]

        st.title(f"Forecast for {result['location']}")

        result_column, gauge_column = st.columns([1, 2])

        with result_column:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if prediction == 1:
                st.error(f"### 🌧️ Rain likely\n**{probability * 100:.1f}%** probability")
            else:
                st.success(f"### ☀️ Rain unlikely\n**{probability * 100:.1f}%** probability")
            st.markdown("</div>", unsafe_allow_html=True)

        with gauge_column:
            gauge_chart = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Rain Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#9d5bff"},
                    "steps": [
                        {"range": [0, 30], "color": "#e6d9ff"},
                        {"range": [30, 60], "color": "#c7a6ff"},
                        {"range": [60, 100], "color": "#7b2ff7"},
                    ],
                },
            ))
            gauge_chart.update_layout(
                height=250, margin=dict(l=20, r=20, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f1ecff"},
            )
            st.plotly_chart(gauge_chart, use_container_width=True)

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔁 Predict again", use_container_width=True):
                go_to("predict_form")
        with c2:
            if st.button("🏠 Back to Home", use_container_width=True):
                go_to("home")

# ---------------------------------------------------------------------------
# PAGE: Model Insights
# ---------------------------------------------------------------------------

elif st.session_state.page == "insights":
    render_back_button("home")
    st.title("Model Performance & Comparison")
    st.write(
        f"Three models were trained and compared. **{metrics['best_model']}** was "
        "selected based on F1-score, which balances precision and recall on this "
        "imbalanced dataset (~78% no-rain days)."
    )

    scores_table = pd.DataFrame(metrics["all_models"])[
        ["name", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    st.dataframe(scores_table.set_index("name"), use_container_width=True)

    long_format_scores = scores_table.melt(id_vars="name", var_name="metric", value_name="score")
    comparison_chart = px.bar(
        long_format_scores, x="metric", y="score", color="name", barmode="group",
        title="Model comparison across metrics",
        color_discrete_sequence=["#9d5bff", "#5b9dff", "#c7a6ff"],
    )
    comparison_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#f1ecff"})
    st.plotly_chart(comparison_chart, use_container_width=True)

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("ROC Curves")
        roc_chart = go.Figure()

        for model_scores in metrics["all_models"]:
            roc_chart.add_trace(go.Scatter(
                x=model_scores["roc_curve"]["fpr"],
                y=model_scores["roc_curve"]["tpr"],
                mode="lines",
                name=f"{model_scores['name']} (AUC={model_scores['roc_auc']:.3f})",
            ))

        roc_chart.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="gray"), name="Random",
        ))
        roc_chart.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#f1ecff"},
        )
        st.plotly_chart(roc_chart, use_container_width=True)

    with right_column:
        st.subheader(f"Confusion Matrix — {metrics['best_model']}")

        best_model_scores = None
        for model_scores in metrics["all_models"]:
            if model_scores["name"] == metrics["best_model"]:
                best_model_scores = model_scores

        confusion = best_model_scores["confusion_matrix"]
        confusion_chart = px.imshow(
            confusion, text_auto=True, color_continuous_scale="Purples",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Rain", "Rain"], y=["No Rain", "Rain"],
        )
        confusion_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f1ecff"})
        st.plotly_chart(confusion_chart, use_container_width=True)

    st.subheader(f"Top Feature Importances — {metrics['best_model']}")
    feature_table = pd.DataFrame(feature_importance)
    feature_chart = px.bar(
        feature_table.sort_values("importance"), x="importance", y="feature", orientation="h",
        color_discrete_sequence=["#7b2ff7"],
    )
    feature_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#f1ecff"})
    st.plotly_chart(feature_chart, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: Data Explorer
# ---------------------------------------------------------------------------

elif st.session_state.page == "explorer":
    render_back_button("home")
    st.title("Explore the Weather Data")
    st.caption("Charts below use an 8,000-row sample of the cleaned training data.")

    purple_blue_scale = ["#1e3c72", "#5b9dff", "#9d5bff", "#c7a6ff"]

    left_column, right_column = st.columns(2)

    with left_column:
        rain_by_month = sample_data.groupby("Month")["RainTomorrow"].mean().reset_index()
        chart = px.bar(
            rain_by_month, x="Month", y="RainTomorrow",
            title="Probability of Rain by Month", labels={"RainTomorrow": "P(Rain)"},
            color_discrete_sequence=["#7b2ff7"],
        )
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#f1ecff"})
        st.plotly_chart(chart, use_container_width=True)

    with right_column:
        chart = px.box(
            sample_data, x="RainTomorrow", y="Humidity3pm",
            title="3pm Humidity vs Next-Day Rain",
            labels={"RainTomorrow": "Rain Tomorrow (0=No, 1=Yes)"},
            color_discrete_sequence=["#5b9dff"],
        )
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#f1ecff"})
        st.plotly_chart(chart, use_container_width=True)

    left_column, right_column = st.columns(2)

    with left_column:
        rain_by_city = (
            sample_data.groupby("Location")["RainTomorrow"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        chart = px.bar(
            rain_by_city, x="RainTomorrow", y="Location", orientation="h",
            title="Top 15 Rainiest Locations (by next-day rain rate)",
            color_discrete_sequence=["#9d5bff"],
        )
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#f1ecff"})
        st.plotly_chart(chart, use_container_width=True)

    with right_column:
        numeric_columns_for_heatmap = [
            "MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm",
            "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
        ]
        correlation_table = sample_data[numeric_columns_for_heatmap].corr()
        chart = px.imshow(
            correlation_table, text_auto=".2f", color_continuous_scale="Purples",
            title="Correlation Heatmap",
        )
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f1ecff"})
        st.plotly_chart(chart, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: About This Project
# ---------------------------------------------------------------------------

elif st.session_state.page == "about":
    render_back_button("home")
    st.title("About This Project")
    st.markdown(f"""
### Rainfall Prediction System

An end-to-end machine learning web app that predicts next-day rainfall
across Australia, built on 10 years of real observational weather data.

**Pipeline:**
1. **Data cleaning & feature engineering** — 142,000+ daily records from 49
   Australian weather stations (2008–2017), missing-value imputation, and
   month-based seasonality features.
2. **Model comparison** — Logistic Regression, Random Forest, and XGBoost
   were trained and evaluated on accuracy, precision, recall, F1, and ROC-AUC.
   **{metrics['best_model']}** was selected as the best performer.
3. **Class imbalance handling** — since ~78% of days have no rain, class
   weighting was used instead of naively optimizing accuracy.
4. **Deployment** — served as an interactive Streamlit web app with a live
   weather auto-fill feature (Open-Meteo API) and full model-transparency
   dashboards (ROC curves, confusion matrix, feature importance).

**Tech stack:** Python, pandas, scikit-learn, XGBoost, Streamlit, Plotly

**Dataset source:** Australian Bureau of Meteorology, via the public
"Rain in Australia" dataset.

---
*This tool is for educational/demonstration purposes and should not be used
for real-world weather-critical decisions — always check an official
forecast (e.g. bom.gov.au) for that.*
""")