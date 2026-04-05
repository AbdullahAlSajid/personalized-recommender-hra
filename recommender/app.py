from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)

# =========================
# Load data
# =========================
exposure = pd.read_csv("data/2025-10-30_exposure.csv")
responses = pd.read_csv("data/2026-01-27_responses.csv")
users = pd.read_csv("data/2025-10-30_users.csv")
user_thetas = pd.read_csv("data/2025-11-14_user_thetas.csv")
item_params = pd.read_csv("data/2025-11-04_item_parameters_Rasch.csv")

try:
    question_texts = pd.read_csv("data/question_texts_texts.csv", encoding="utf-8")
except UnicodeDecodeError:
    question_texts = pd.read_csv("data/question_texts_texts.csv", encoding="utf-8-sig")

# =========================
# Normalize key columns
# =========================
if "student_id" in responses.columns:
    responses["student_id"] = responses["student_id"].astype(str)

if "sanity_text_id" in responses.columns:
    responses["sanity_text_id"] = responses["sanity_text_id"].astype(str)

if "sanity_text_id" in item_params.columns:
    item_params["sanity_text_id"] = item_params["sanity_text_id"].astype(str)

if "sanity_text_id" in exposure.columns:
    exposure["sanity_text_id"] = exposure["sanity_text_id"].astype(str)

if "sanity_text_id" in question_texts.columns:
    question_texts["sanity_text_id"] = question_texts["sanity_text_id"].astype(str)

if "student_id" in users.columns:
    users["student_id"] = users["student_id"].astype(str)

if "user_id" in user_thetas.columns:
    user_thetas["user_id"] = user_thetas["user_id"].astype(str)

# =========================
# Prepare recommendation data
# =========================

# Text difficulty per text
text_difficulty = item_params.groupby("sanity_text_id")["b"].mean().reset_index()
text_difficulty.rename(columns={"b": "text_difficulty"}, inplace=True)

# Merge in metadata from exposure if available
available_exposure_columns = [
    col for col in ["sanity_text_id", "title", "serialNumber"] if col in exposure.columns
]
if available_exposure_columns:
    text_difficulty = text_difficulty.merge(
        exposure[available_exposure_columns].drop_duplicates(),
        on="sanity_text_id",
        how="left"
    )

# Merge actual text data from question_texts_texts.csv
text_preview_df = question_texts[["sanity_text_id", "serialNumber", "title", "body"]].drop_duplicates().copy()

text_difficulty = text_difficulty.merge(
    text_preview_df,
    on="sanity_text_id",
    how="left",
    suffixes=("", "_qt")
)

# Prefer question_texts values when missing
if "title_qt" in text_difficulty.columns:
    if "title" not in text_difficulty.columns:
        text_difficulty["title"] = text_difficulty["title_qt"]
    else:
        text_difficulty["title"] = text_difficulty["title"].fillna(text_difficulty["title_qt"])
    text_difficulty.drop(columns=["title_qt"], inplace=True)

if "serialNumber_qt" in text_difficulty.columns:
    if "serialNumber" not in text_difficulty.columns:
        text_difficulty["serialNumber"] = text_difficulty["serialNumber_qt"]
    else:
        text_difficulty["serialNumber"] = text_difficulty["serialNumber"].fillna(text_difficulty["serialNumber_qt"])
    text_difficulty.drop(columns=["serialNumber_qt"], inplace=True)


def make_preview(value, length=220):
    if pd.isna(value):
        return None

    text = str(value)
    lines = text.splitlines()
    lines = [line for line in lines if not line.strip().startswith("![")]

    cleaned = " ".join(lines)
    cleaned = cleaned.replace("###", "").replace("##", "").replace("#", "")
    cleaned = " ".join(cleaned.split())

    if len(cleaned) <= length:
        return cleaned

    return cleaned[:length].rstrip() + "..."


text_difficulty["preview_text"] = text_difficulty["body"].apply(make_preview)

# User theta
if "Rasch" in user_thetas.columns:
    user_theta = user_thetas[["user_id", "Rasch"]].rename(columns={"Rasch": "theta"})
elif "theta" in user_thetas.columns:
    user_theta = user_thetas[["user_id", "theta"]].copy()
else:
    user_theta = pd.DataFrame(columns=["user_id", "theta"])

# Seen texts per user
if "student_id" in responses.columns and "sanity_text_id" in responses.columns:
    user_seen = responses.groupby("student_id")["sanity_text_id"].unique().to_dict()
else:
    user_seen = {}

# KNN for difficulty-based recommendations
if not text_difficulty.empty:
    X = text_difficulty[["text_difficulty"]].fillna(0).values
    knn = NearestNeighbors(n_neighbors=min(20, len(X)))
    knn.fit(X)
else:
    knn = None

# Popularity from responses
if "sanity_text_id" in responses.columns:
    text_popularity = (
        responses.groupby("sanity_text_id")
        .size()
        .reset_index(name="popularity_count")
        .sort_values("popularity_count", ascending=False)
    )
else:
    text_popularity = pd.DataFrame(columns=["sanity_text_id", "popularity_count"])

# Join popularity with text metadata
popularity_texts = text_popularity.merge(
    text_difficulty[["sanity_text_id", "serialNumber", "title", "body", "preview_text"]].drop_duplicates(),
    on="sanity_text_id",
    how="left"
)

# =========================
# Recommendation functions
# =========================
def get_difficulty_recommendations(user_id, k=2):
    if knn is None or user_theta.empty:
        return []

    user_id = str(user_id)
    theta_row = user_theta.loc[user_theta["user_id"] == user_id, "theta"]

    if theta_row.empty:
        return []

    theta = theta_row.values[0]
    seen = user_seen.get(user_id, [])

    distances, indices = knn.kneighbors([[theta]])
    recs = text_difficulty.iloc[indices[0]].copy()

    if "sanity_text_id" in recs.columns:
        recs["sanity_text_id"] = recs["sanity_text_id"].astype(str)
        recs = recs[~recs["sanity_text_id"].isin(seen)]

    recs = recs.head(k).replace({np.nan: None})

    keep_columns = [
        col for col in [
            "sanity_text_id",
            "serialNumber",
            "title",
            "preview_text",
            "body"
        ] if col in recs.columns
    ]

    return recs[keep_columns].to_dict(orient="records")


def get_popularity_recommendations(user_id, k=2):
    if popularity_texts.empty:
        return []

    user_id = str(user_id)
    seen = user_seen.get(user_id, [])

    recs = popularity_texts.copy()
    recs["sanity_text_id"] = recs["sanity_text_id"].astype(str)

    if seen:
        recs = recs[~recs["sanity_text_id"].isin(seen)]

    recs = recs.head(k).replace({np.nan: None})

    keep_columns = [
        col for col in [
            "sanity_text_id",
            "serialNumber",
            "title",
            "preview_text",
            "body",
            "popularity_count"
        ] if col in recs.columns
    ]

    return recs[keep_columns].to_dict(orient="records")


def get_random_recommendations(user_id, k=2):
    return []

# =========================
# Routes
# =========================
@app.get("/")
def home():
    return "Backend is running"


@app.post("/api/login")
def login():
    body = request.get_json()
    student_id = str(body.get("student_id", "")).strip()

    if not student_id:
        return jsonify({
            "success": False,
            "message": "student_id is required"
        }), 400

    if "user_id" not in users.columns:
        return jsonify({
            "success": False,
            "message": "Column 'user_id' not found in users.csv"
        }), 500

    # Map student_id to user_id for lookup
    user_row = users[users["user_id"].astype(str) == student_id]

    if user_row.empty:
        return jsonify({
            "success": False,
            "message": "User not found"
        }), 404

    user = user_row.iloc[0].replace({np.nan: None}).to_dict()

    difficulty_recommendations = get_difficulty_recommendations(student_id, k=2)
    popularity_recommendations = get_popularity_recommendations(student_id, k=2)
    random_recommendations = get_random_recommendations(student_id, k=2)

    return jsonify({
        "success": True,
        "user": user,
        "recommendations": {
            "difficulty": difficulty_recommendations,
            "popularity": popularity_recommendations,
            "random": random_recommendations
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)