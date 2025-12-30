import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

MODEL_PATH = "models/knn_model.pkl"
DATA_PATH = "data/characters.csv"
QUESTIONS_PATH = "data/questions.json"

model_data = joblib.load(MODEL_PATH)
scaler = model_data["scaler"]
TRAITS = model_data["traits"]

df = pd.read_csv(DATA_PATH)

with open(QUESTIONS_PATH, "r") as f:
    questions_data = json.load(f)

universes = sorted(df["universe"].unique())

print("Choose a universe:")
for i, u in enumerate(universes, 1):
    print(f"{i}. {u}")

u_choice = int(input("Enter choice: "))
selected_universe = universes[u_choice - 1]

trait_scores = {t: 0 for t in TRAITS}

for q in questions_data["questions"]:
    print("\n" + q["question"])
    options = list(q["options"].keys())

    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")

    choice = int(input("Choose option: "))
    impacts = q["options"][options[choice - 1]]

    for trait, value in impacts.items():
        trait_scores[trait] += value

user_vector = np.array([trait_scores[t] for t in TRAITS]).reshape(1, -1)
user_vector = scaler.transform(user_vector)

df_u = df[df["universe"] == selected_universe].reset_index(drop=True)
X_u = scaler.transform(df_u[TRAITS])

distances = pairwise_distances(user_vector, X_u, metric="euclidean")[0]

df_u["distance"] = distances
df_u = df_u.sort_values("distance")

print("\nYour closest matches:")
top_matches = df_u.head(3)

for i, row in enumerate(top_matches.itertuples(), 1):
    if pd.isna(row.movie) or row.movie == "":
        print(f"{i}. {row.name}")
    else:
        print(f"{i}. {row.name} ({row.movie})")