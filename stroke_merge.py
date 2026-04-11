import pandas as pd
import numpy as np

df_framingham = pd.read_csv('data/24-framingham.csv')
df_framingham.head()
df_stroke = pd.read_csv('data/26-stroke.csv')
df_stroke.head()

df_framingham["gender"]            = df_framingham["Sex"].str.capitalize()
df_framingham["hypertension"]      = df_framingham["prevalentHyp"].astype(int)
df_framingham["heart_disease"]     = df_framingham["TenYearCHD"].astype(int)
df_framingham["stroke"]            = df_framingham["prevalentStroke"].astype(int)
df_framingham["avg_glucose_level"] = df_framingham["glucose"]
df_framingham["smoking_status"] = df_framingham["currentSmoker"].map({"Yes": 1, "No": 0})

df_framingham["ever_married"]   = np.nan
df_framingham["work_type"]      = np.nan
df_framingham["Residence_type"] = np.nan
df_stroke["smoking_status"] = df_stroke["smoking_status"].map({
    "smokes":          1,
    "formerly smoked": 0,
    "never smoked":    0,
    "Unknown":         np.nan
})

bonus_cols = ["education", "cigsPerDay", "BPMeds", "totChol",
              "sysBP", "diaBP", "heartRate"]

df_framingham_aligned = df_framingham[[
    "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "BMI", "smoking_status", "stroke",
    *bonus_cols
]].rename(columns={"BMI": "bmi"})

df_stroke["source"]               = "stroke"
df_framingham_aligned["source"]   = "framingham"

for col in bonus_cols:
    df_stroke[col] = np.nan

df_stroke_aligned = df_stroke[[
    "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke",
    *bonus_cols, "source"
]]

merged = pd.concat([df_stroke_aligned, df_framingham_aligned], ignore_index=True)
merged["hypertension"]    = merged["hypertension"].astype("Int64")
merged["heart_disease"]   = merged["heart_disease"].astype("Int64")
merged["stroke"]          = merged["stroke"].astype("Int64")
merged["smoking_status"]  = merged["smoking_status"].astype("Int64")
merged["age"]             = merged["age"].astype(float)

for col in ["bmi", "avg_glucose_level", "cigsPerDay",
            "totChol", "sysBP", "diaBP", "heartRate"]:
    merged[col] = merged.groupby("source")[col].transform(
        lambda x: x.fillna(x.median())
    )

for col in ["smoking_status", "BPMeds", "education",
            "ever_married", "work_type", "Residence_type"]:
    merged[col] = merged.groupby("source")[col].transform(
        lambda x: x.fillna(x.mode()[0]) if x.notna().any() else x
    )