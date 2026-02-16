import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("Salary Data.csv")

df = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Salary']]
df = df.dropna()

# -----------------------------
# ENCODING
# -----------------------------
le_gender = LabelEncoder()
le_edu = LabelEncoder()
le_job = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Education Level'] = le_edu.fit_transform(df['Education Level'])
df['Job Title'] = le_job.fit_transform(df['Job Title'])

# -----------------------------
# FEATURES & TARGET
# -----------------------------
feature_names = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
X = df[feature_names]
y = df['Salary']

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODELS
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

results = {}

print("\nTraining Models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {
        "model": model,
        "mae": mae,
        "r2": r2
    }

    print(f"{name}")
    print(f"MAE: {round(mae,2)}")
    print(f"R2: {round(r2,4)}")
    print("----------------------")

# -----------------------------
# PICK BEST MODEL
# -----------------------------
best_model_name = max(results, key=lambda x: results[x]["r2"])
best_model = results[best_model_name]["model"]

print("\nBest Model Selected:", best_model_name)
print("R2 Score:", round(results[best_model_name]["r2"],4))

# -----------------------------
# CROSS VALIDATION
# -----------------------------
print("\nRunning Cross Validation...")

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Cross Validation R2 Scores:", cv_scores)
print("Average CV R2:", round(cv_scores.mean(),4))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
print("\nFeature Importance:")

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    for name, imp in zip(feature_names, importances):
        print(f"{name}: {round(imp * 100, 2)}%")
else:
    print("Feature importance not supported for this model")

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(best_model, "model.pkl")

print("\nModel saved as model.pkl")
