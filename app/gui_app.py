
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import pairwise_distances
import base64

MODEL_PATH = "models/knn_model.pkl"
DATA_PATH = "data/characters.csv"
QUESTIONS_PATH = "data/questions.json"
BG_IMAGE = "assets/background.jpg"

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_base64 = get_base64_image(BG_IMAGE)

st.set_page_config(
    page_title="Who Are You?",
    page_icon="✨",
    layout="centered"
)

st.markdown(f"""
<style>
.stApp {{
    background-color: #fde7ef;
    background-image: url("data:image/jpg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(253, 231, 239, 0.75);
    z-index: -1;
}}

.block-container {{
    background: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 22px;
}}

div[role="radiogroup"] > label {{
    margin-bottom: 6px;
}}

.stButton > button {{
    background-color: #f48fb1 !important;  /* soft pink */
    color: #ffffff !important;
    border-radius: 14px;
    border: none;
    padding: 0.6em 1.2em;
    font-weight: 600;
}}

.stButton > button:hover {{
    background-color: #ec6f9f !important;
}}

.stMarkdown,
.stText,
p,
h1, h2, h3, h4, h5, h6,
label:not([data-baseweb="select"]),
span:not([data-baseweb="select"]) {{
    color: #000000 !important;
}}
</style>
""", unsafe_allow_html=True)

model_data = joblib.load(MODEL_PATH)
scaler = model_data["scaler"]
TRAITS = model_data["traits"]

df = pd.read_csv(DATA_PATH)

with open(QUESTIONS_PATH, "r") as f:
    questions_data = json.load(f)

st.title("✨ Which Fictional Character Are You?")
st.caption("Answer a few questions and discover your closest match")

universe = st.selectbox(
    "Choose a universe",
    ["Harry Potter", "Disney", "Pixar", "Disney Princesses 👑"]
)

trait_scores = {t: 0 for t in TRAITS}

st.divider()

for q in questions_data["questions"]:
    choice = st.radio(
        q["question"],
        list(q["options"].keys()),
        key=q["id"]
    )
    for trait, value in q["options"][choice].items():
        trait_scores[trait] += value

if st.button("✨ Reveal My Character"):
    user_vector = np.array(
        [trait_scores[t] for t in TRAITS]
    ).reshape(1, -1)

    user_vector = scaler.transform(user_vector)

    df_u = df[df["universe"] == universe].reset_index(drop=True)
    X_u = scaler.transform(df_u[TRAITS])

    distances = pairwise_distances(user_vector, X_u)[0]
    df_u["distance"] = distances
    top = df_u.sort_values("distance").iloc[0]

    st.divider()

    if pd.isna(top["movie"]) or top["movie"] == "":
        st.subheader(f"✨ You are most like **{top['name']}**")
    else:
        st.subheader(
            f"✨ You are most like **{top['name']}**  \n*({top['movie']})*"
        )