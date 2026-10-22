import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

DATA_FILE = "salary_data.csv"
MODEL_FILE = "salary_model.pkl"
FEATURES = ["Experience", "Education", "Age", "Skills", "JobRole"]
TARGET = "Salary"

def make_demo_dataset():
    # Synthetic demonstration data only. Replace with reliable real-world data for meaningful predictions.
    rows = [
        [0, "UG", 21, "Python", "Software Engineer", 250000],
        [1, "UG", 22, "Python", "Software Engineer", 320000],
        [2, "UG", 23, "C++", "Embedded Engineer", 350000],
        [3, "UG", 24, "Java", "Software Engineer", 400000],
        [4, "PG", 26, "Python", "Data Analyst", 550000],
        [5, "PG", 27, "Data Science", "Data Analyst", 650000],
        [6, "PG", 28, "Python", "Software Engineer", 700000],
        [7, "UG", 29, "Java", "Embedded Engineer", 600000],
        [8, "PG", 31, "Data Science", "Data Analyst", 850000],
        [9, "PG", 32, "C++", "Embedded Engineer", 750000],
        [10, "PG", 34, "Python", "Software Engineer", 900000],
        [12, "PG", 36, "Data Science", "Data Analyst", 1050000],
        [14, "PG", 39, "C++", "Embedded Engineer", 1200000],
        [2, "UG", 24, "Java", "Data Analyst", 380000],
        [5, "PG", 29, "Python", "Embedded Engineer", 620000],
    ]
    return pd.DataFrame(rows, columns=FEATURES + [TARGET])

def main():
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        df = make_demo_dataset()
        df.to_csv(DATA_FILE, index=False)
        print(f"{DATA_FILE} did not exist, so a synthetic demo file was created.")

    # Backward compatibility with the user's original dataset
    if "JobRole" not in df.columns:
        df["JobRole"] = "Software Engineer"

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {DATA_FILE}: {missing}")

    df = df[required].dropna().drop_duplicates()
    if len(df) < 5:
        raise ValueError("Please provide at least 5 valid records; 20+ records is preferable for a demo.")

    for col in ["Experience", "Age", "Salary"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Experience", "Age", "Salary"])
    if len(df) < 5:
        raise ValueError("Not enough numeric records after cleaning.")

    df.to_csv(DATA_FILE, index=False)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    categorical = ["Education", "Skills", "JobRole"]
    preprocessor = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical)
    ], remainder="passthrough")

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    results = []
    trained = {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("regressor", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            "Model": name,
            "MAE (₹)": round(mean_absolute_error(y_test, pred), 2),
            "RMSE (₹)": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
            "R² Score": round(r2_score(y_test, pred), 3) if len(y_test) >= 2 else float("nan")
        })
        trained[name] = pipe

    # Random Forest remains the main model for the prediction page
    joblib.dump(trained["Random Forest"], MODEL_FILE)
    pd.DataFrame(results).to_csv("model_comparison.csv", index=False)

    print("Training completed.")
    print(f"Records used: {len(df)}")
    print("\nModel comparison:")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSaved model to {MODEL_FILE}")
    print("Reminder: metrics from a tiny synthetic dataset are only a demonstration.")

if __name__ == "__main__":
    main()
