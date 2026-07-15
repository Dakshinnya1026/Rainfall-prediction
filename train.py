"""
Rainfall Prediction - Model Training Script (Simplified / Beginner Version)

What this file does, in plain steps:
1. Load the weather CSV file
2. Clean it up (fix missing values, convert Yes/No to 1/0)
3. Split the data into a training part and a testing part
4. Train 3 different models on the training part
5. Test all 3 models and see which one is best
6. Save the best model, plus some helper info, to the "models" folder
"""

import json
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Hide harmless warning messages so the output stays clean
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# SETTINGS - the file paths and column names we will use everywhere below
# ---------------------------------------------------------------------------

DATA_PATH = "data/weatherAUS.csv"
MODEL_FOLDER = "models"

# These are the columns that already contain NUMBERS
NUMBER_COLUMNS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
    "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm", "Month",
]

# These are the columns that contain TEXT / CATEGORIES
TEXT_COLUMNS = [
    "Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday",
]

# This is the column we are trying to predict
TARGET_COLUMN = "RainTomorrow"


def load_and_clean_data():
    """Reads the CSV and fixes it up so it's ready for training."""

    # Step 1: read the raw file into a table
    data = pd.read_csv(DATA_PATH)

    # Step 2: throw away any row where we don't know the correct answer
    data = data.dropna(subset=[TARGET_COLUMN])

    # Step 3: turn the Date column into a Month number (1-12)
    # We do this because rain patterns change by season/month.
    data["Date"] = pd.to_datetime(data["Date"])
    data["Month"] = data["Date"].dt.month

    # Step 4: turn "Yes"/"No" text into 1/0 numbers
    data["RainToday"] = data["RainToday"].map({"Yes": 1, "No": 0})
    data[TARGET_COLUMN] = data[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # Step 5: we don't need the raw Date column anymore
    if "Date" in data.columns:
        data = data.drop(columns=["Date"])

    return data


def build_pipeline(model):
    """
    Builds a 'recipe' that:
    - fills in missing numbers with the median
    - scales numbers so they're all a similar size
    - fills in missing categories with the most common one
    - turns categories into 0/1 columns (one-hot encoding)
    - then finally runs the given model
    """

    # Recipe for number columns
    number_steps = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Recipe for text/category columns
    text_steps = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("to_numbers", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Combine both recipes, telling it which columns get which recipe
    preprocessing = ColumnTransformer(transformers=[
        ("numbers", number_steps, NUMBER_COLUMNS),
        ("text", text_steps, TEXT_COLUMNS),
    ])

    # Chain: clean the data first, THEN run the model
    full_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessing),
        ("model", model),
    ])
    return full_pipeline


def score_model(model_name, pipeline, X_test, y_test):
    """Runs the model on test data and returns a dictionary of scores."""

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]  # chance of "Yes rain"

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)

    result = {
        "name": model_name,
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "f1": round(f1_score(y_test, predictions), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "roc_curve": {
            "fpr": false_positive_rate[::5].tolist(),
            "tpr": true_positive_rate[::5].tolist(),
        },
    }

    print(
        f"{model_name:20s} | Accuracy {result['accuracy']:.3f} "
        f"| Precision {result['precision']:.3f} | Recall {result['recall']:.3f} "
        f"| F1 {result['f1']:.3f} | ROC-AUC {result['roc_auc']:.3f}"
    )
    return result


def get_top_features(pipeline, how_many=15):
    """Finds which input columns mattered most to the model's decisions."""

    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None  # Logistic Regression doesn't support this

    # Get the column names AFTER one-hot encoding (e.g. Location_Sydney)
    text_column_names = (
        pipeline.named_steps["preprocessing"]
        .named_transformers_["text"]
        .named_steps["to_numbers"]
        .get_feature_names_out(TEXT_COLUMNS)
    )
    all_column_names = NUMBER_COLUMNS + list(text_column_names)

    # Pair up each column name with its importance score
    pairs = []
    for name, score in zip(all_column_names, model.feature_importances_):
        pairs.append({"feature": name, "importance": round(float(score), 4)})

    # Sort so the most important features come first
    pairs.sort(key=lambda item: item["importance"], reverse=True)

    # Keep only the top N
    return pairs[:how_many]


def main():
    print("Step 1: Loading and cleaning the data...")
    data = load_and_clean_data()
    print(f"   -> {len(data)} rows ready to use")

    # X = the inputs, y = the answer we want to predict
    X = data[NUMBER_COLUMNS + TEXT_COLUMNS]
    y = data[TARGET_COLUMN]

    print("Step 2: Splitting into training data and testing data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # There are more "no rain" days than "rain" days.
    # This number tells XGBoost how much extra attention to give rain days.
    no_rain_count = (y_train == 0).sum()
    rain_count = (y_train == 1).sum()
    imbalance_ratio = no_rain_count / rain_count

    print("Step 3: Training 3 different models...")
    models_to_try = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=imbalance_ratio, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
    }

    all_scores = []
    trained_pipelines = {}

    for model_name, model in models_to_try.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        scores = score_model(model_name, pipeline, X_test, y_test)
        all_scores.append(scores)
        trained_pipelines[model_name] = pipeline

    print("Step 4: Picking the best model (highest F1 score)...")
    best_score_so_far = -1
    best_model_name = None
    for scores in all_scores:
        if scores["f1"] > best_score_so_far:
            best_score_so_far = scores["f1"]
            best_model_name = scores["name"]

    best_pipeline = trained_pipelines[best_model_name]
    print(f"   -> Best model: {best_model_name} (F1 = {best_score_so_far})")

    print("Step 5: Saving everything the app needs...")

    # The trained model itself
    joblib.dump(best_pipeline, f"{MODEL_FOLDER}/best_pipeline.joblib")

    # The scores for all 3 models, so the app can show a comparison table
    with open(f"{MODEL_FOLDER}/metrics.json", "w") as f:
        json.dump({"all_models": all_scores, "best_model": best_model_name}, f, indent=2)

    # Which features mattered most
    top_features = get_top_features(best_pipeline)
    with open(f"{MODEL_FOLDER}/feature_importance.json", "w") as f:
        json.dump(top_features, f, indent=2)

    # Dropdown menu options for the app (list of cities, list of wind directions)
    city_list = sorted(data["Location"].unique().tolist())
    with open(f"{MODEL_FOLDER}/locations.json", "w") as f:
        json.dump(city_list, f)

    wind_direction_list = sorted(data["WindGustDir"].dropna().unique().tolist())
    with open(f"{MODEL_FOLDER}/wind_dirs.json", "w") as f:
        json.dump(wind_direction_list, f)

    # A small sample of data, just for drawing charts quickly in the app
    sample = data.sample(n=8000, random_state=42)
    sample.to_csv(f"{MODEL_FOLDER}/sample_for_viz.csv", index=False)

    print("Done! All files saved inside the 'models' folder.")


# This makes sure main() only runs when you type "python train.py" directly
if __name__ == "__main__":
    main()
