"""
build_pipeline.py
------------------
Fits a StandardScaler on placementdata.csv using the exact feature order
the uploaded LogisticRegression model expects, then bundles it together
with feature stats (used to silently fill the near-irrelevant StudentID
field) into pipeline_bundle.pkl for the Streamlit app to use.

Run once:
    python build_pipeline.py
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "StudentID",
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks",
]

df = pd.read_csv("placementdata.csv")
df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map({"No": 0, "Yes": 1})
df["PlacementTraining"] = df["PlacementTraining"].map({"No": 0, "Yes": 1})

X = df[FEATURE_COLS]

scaler = StandardScaler()
scaler.fit(X)

student_id_mean = float(df["StudentID"].mean())

model = joblib.load("placement_model.pkl")

bundle = {
    "model": model,
    "scaler": scaler,
    "feature_cols": FEATURE_COLS,
    "student_id_fill": student_id_mean,
}

joblib.dump(bundle, "pipeline_bundle.pkl")
print("Saved pipeline_bundle.pkl")
print("StudentID auto-fill value:", student_id_mean)
