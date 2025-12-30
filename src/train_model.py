import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# CONFIG
DATA_PATH = "data/characters.csv"
MODEL_PATH = "models/knn_model.pkl"

TRAITS = [
    "emotional_control",
    "empathy",
    "optimism",
    "anger_intensity",
    "intelligence",
    "creativity",
    "strategic_thinking",
    "problem_solving",
    "leadership",
    "teamwork",
    "communication",
    "sense_of_justice",
    "loyalty",
    "bravery",
    "impulsiveness"
]

K = 3   # good balance for small-medium datasets

# LOAD DATA
df = pd.read_csv(DATA_PATH)

X = df[TRAITS]
y = df["name"]

# SCALE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# TRAIN MODEL
knn = KNeighborsClassifier(
    n_neighbors=K,
    metric="euclidean"
)

knn.fit(X_scaled, y)

# SAVE MODEL + SCALER
joblib.dump(
    {
        "model": knn,
        "scaler": scaler,
        "traits": TRAITS
    },
    MODEL_PATH
)

print("✅ KNN model trained and saved successfully.")
print(f"📦 Characters trained on: {len(df)}")
