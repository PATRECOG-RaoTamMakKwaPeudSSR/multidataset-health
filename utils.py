# ===========================================================================
# PREPROCESS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle

RANDOM_STATE = 42
TEST_SIZE = 0.2

def preprocess_statlog(df: pd.DataFrame, scale: bool) -> tuple:
    df_new = df.copy()
    
    df_new.attrs['dataset_name'] = 'Statlog'
    
    # Step 1: Rename columns
    df_new.rename(columns={
        "chest-pain": "cp_type",
        "rest-bp": "sbp",
        "serum-chol": "chol",
        "fasting-blood-sugar": "fbs",
        "electrocardiographic": "ecg",
        'max-heart-rate': 'hr',
        "major-vessels": "mv",
        "heart-disease": "target"
    }, inplace=True)
    
    # Step 2: Cast to int
    for col in ["age", "sex", "cp_type", "sbp", "chol", "fbs", "ecg", "hr", "angina", "slope", "mv", "thal", "target"]:
        df_new[col] = df_new[col].astype(int)
    
    # Step 3: Map values
    df_new['thal'] = df_new['thal'].map({3: 0, 6: 1, 7: 2})
    df_new['target'] = df_new['target'].map({1: 0, 2: 1})
    
    # Step 4: Drop duplicates
    df_new.drop_duplicates(inplace=True)
    
    # Step 5: Scale continuous features to [0, 1]
    if not scale:
        return df_new, None
    
    cont_cols = ['age', 'sbp', 'chol', 'hr', 'oldpeak']
    scaler = MinMaxScaler()
    df_new[cont_cols] = scaler.fit_transform(df_new[cont_cols])
    
    return df_new, scaler

def preprocess_chd(df: pd.DataFrame, scale: bool) -> tuple:
    df_new = df.copy()
    
    df_new.attrs['dataset_name'] = 'CHD'
        
    # Step 1: Map values
    df_new['famhist'] = df_new['famhist'].map({'Present': 1, 'Absent': 0})

    # Step 2: Rename columns
    df_new.rename(columns={
        'obesity': 'bmi',
        'chd': 'target'
    }, inplace=True)
    
    # Step 3: Drop duplicates
    df_new.drop_duplicates(inplace=True)
    
    # Step 4: Scale continuous features to [0, 1]
    if not scale:
        return df_new, None
    
    cont_cols = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'bmi', 'alcohol', 'age']
    scaler = MinMaxScaler()
    df_new[cont_cols] = scaler.fit_transform(df_new[cont_cols])
    
    return df_new, scaler

def preprocess_framingham(df: pd.DataFrame, scale: bool) -> tuple:
    df_new: pd.DataFrame = df.copy()
    
    df_new.attrs['dataset_name'] = 'Framingham'
    
    # Step 1: Rename columns
    df_new.rename(columns={
        "Sex": "sex",
        "age": "age",
        "education": "edu",
        "currentSmoker": "smoking_status",
        "cigsPerDay": "num_cigs_per_day",
        "BPMeds": "bp_status",
        "prevalentStroke": "stroke_status",
        "prevalentHyp": "hypertension_status",
        "diabetes": "diabetes_status",
        "totChol": "chol",
        "sysBP": "sbp",
        "diaBP": "dbp",
        "BMI": "bmi",
        "heartRate": "hr",
        "glucose": "glucose_level",
        "TenYearCHD": "target"
    }, inplace=True)
    
    # Step 2: Map categorical values
    df_new['sex'] = df_new['sex'].map({"male": 1, "female": 0})
    df_new['smoking_status'] = df_new['smoking_status'].map({"Yes": 1, "No": 0})
    df_new['diabetes_status'] = df_new['diabetes_status'].map({"Yes": 1, "No": 0})
    
    # Step 3: Handle missing values
    df_new['edu'] = df_new['edu'].fillna(df_new['edu'].mode()[0])
    df_new['num_cigs_per_day'] = df_new['num_cigs_per_day'].fillna(df_new['num_cigs_per_day'].median())
    df_new["bp_status"] = df_new["bp_status"].fillna(df_new["bp_status"].mode()[0])
    df_new["chol"] = df_new["chol"].fillna(df_new["chol"].mean())
    df_new["bmi"] = df_new["bmi"].fillna(df_new["bmi"].mean())
    df_new["hr"] = df_new["hr"].fillna(df_new["hr"].mean())
    df_new["glucose_level"] = df_new["glucose_level"].fillna(df_new["glucose_level"].mean())
    
    # Step 4: Convert types
    df_new['edu'] = df_new['edu'].astype(int)
    df_new['bp_status'] = df_new['bp_status'].astype(int)
    df_new['stroke_status'] = df_new['stroke_status'].astype(int)
    df_new['hypertension_status'] = df_new['hypertension_status'].astype(int)
    df_new['smoking_status'] = df_new['smoking_status'].astype(int)
    df_new["target"] = df_new["target"].astype(int)

    # Step 5: Drop duplicates
    df_new.drop_duplicates(inplace=True)
    
    # Step 6: Scale continuous features to [0, 1]
    if not scale:
        return df_new, None
    
    cont_cols = ['age', 'num_cigs_per_day', 'chol', 'sbp', 'dbp', 'bmi', 'hr', 'glucose_level']
    scaler = MinMaxScaler()
    df_new[cont_cols] = scaler.fit_transform(df_new[cont_cols])
    
    return df_new, scaler

def preprocess_heart(df: pd.DataFrame, scale: bool) -> tuple:
    df_new = df.copy()
    
    df_new.attrs['dataset_name'] = 'Heart'
    
    # Step 1: Rename columns
    df_new.rename(columns={
        "cp": "cp_type",
        "trestbps": "sbp",
        "restecg": "ecg",
        "thalach": "hr",
        "exang": "angina",
        "ca": "mv"
    }, inplace=True)
    
    # Step 2: Map values
    df_new['cp_type'] = df_new['cp_type'].map({0: 1, 1: 2, 2: 3, 3: 4})
    df_new['slope'] = df_new['slope'].map({0: 1, 1: 2, 2: 3})

    # Step 3: Drop duplicates
    df_new.drop_duplicates(inplace=True)

    # Step 4: Scale continuous features to [0, 1]
    if not scale:
        return df_new, None
    
    cont_cols = ['age', 'sbp', 'chol', 'hr', 'oldpeak']
    scaler = MinMaxScaler()
    df_new[cont_cols] = scaler.fit_transform(df_new[cont_cols])
    
    return df_new, scaler

def preprocess_stroke(df: pd.DataFrame, scale: bool) -> tuple:
    df_new: pd.DataFrame = df.copy()
    
    df_new.attrs['dataset_name'] = 'Stroke'
    
    # Step 0: Drop 'id' column
    df_new.drop(columns=['id'], inplace=True)
    
    # Step 1: Rename columns
    df_new.rename(columns={
        "gender": "sex",
        "age": "age",
        "hypertension": "hypertension_status",
        "heart_disease": "heart_disease_status",
        "ever_married": "marital_status",
        "work_type": "work_type",
        "Residence_type": "residence_type",
        "avg_glucose_level": "glucose_level",
        "bmi": "bmi",
        "smoking_status": "smoking_status",
        "stroke": "target"
    }, inplace=True)
    
    # Step 2: Map categorical values
    df_new['sex'] = df_new['sex'].map({"Male": 1, "Female": 0})
    df_new["marital_status"] = df_new["marital_status"].map({"Yes": 1, "No": 0})
    df_new["work_type"] = df_new["work_type"].map({"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4})
    df_new["residence_type"] = df_new["residence_type"].map({"Urban": 1, "Rural": 0})
    df_new['smoking_status'] = df_new['smoking_status'].map({"formerly smoked": 0, "never smoked": 1, "smokes": 2, "Unknown": 3})

    df_new.dropna(subset=['sex'], inplace=True) # Others gender
    
    # Step 3: Convert types
    df_new['sex'] = df_new['sex'].astype(int)
    
    # Step 4: Handle missing values
    df_new["bmi"] = df_new["bmi"].fillna(df_new["bmi"].mean())

    # Step 5: Drop duplicates
    df_new.drop_duplicates(inplace=True)
    
    # Step 6: Scale continuous features to [0, 1]
    if not scale:
        return df_new, None
    
    cont_cols = ["age", "glucose_level", "bmi"]
    scaler = MinMaxScaler()
    df_new[cont_cols] = scaler.fit_transform(df_new[cont_cols])
    
    return df_new, scaler

# =====
# Pete
def preprocess_statlog_framingham_stroke_intersection(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_new = df.copy()
    scaler = MinMaxScaler()
    df_new[numeric_cols] = scaler.fit_transform(df_new[numeric_cols])
    return df_new, scaler

def preprocess_statlog_framingham_stroke_union(df: pd.DataFrame):
    df_new = df.copy()
    drop_cols = ["cp_type", "fbs", "ecg", "angina", "oldpeak", "slope", "mv", "thal", "smoking_status", "hypertension_status", "bmi", "glucose_level"]
    df_new = df_new.drop(columns=drop_cols)
    mode_cols = ["bp_status", "stroke_status", "diabetes_status", "heart_disease_status", "marital_status", "residence_type", "work_type"]
    df_new[mode_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df_new[mode_cols])
    med_cols = ["edu", "num_cigs_per_day"]
    df_new[med_cols] = SimpleImputer(strategy="median").fit_transform(df_new[med_cols])
    mean_cols = ["sbp", "chol", "hr", "dbp"]
    df_new[mean_cols] = SimpleImputer(strategy="median").fit_transform(df_new[mean_cols])  # ใช้ median imputer ตาม pete.ipynb
    cols = df_new.select_dtypes(include=["number"]).columns
    scaler = MinMaxScaler()
    df_new[cols] = scaler.fit_transform(df_new[cols])
    return df_new, scaler

def preprocess_framingham_stroke_intersection(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_new = df.copy()
    scaler = MinMaxScaler()
    df_new[numeric_cols] = scaler.fit_transform(df_new[numeric_cols])
    return df_new, scaler

def preprocess_framingham_stroke_union(df: pd.DataFrame):
    df_new = df.copy()
    mode_cols = ["bp_status", "stroke_status", "diabetes_status", "heart_disease_status", "marital_status", "work_type", "residence_type"]
    df_new[mode_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df_new[mode_cols])
    med_cols = ["edu", "num_cigs_per_day", "chol"]
    df_new[med_cols] = SimpleImputer(strategy="median").fit_transform(df_new[med_cols])
    mean_cols = ["sbp", "dbp", "hr"]
    df_new[mean_cols] = SimpleImputer(strategy="mean").fit_transform(df_new[mean_cols])
    cols = df_new.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    df_new[cols] = scaler.fit_transform(df_new[cols])
    return df_new, scaler

def preprocess_statlog_heart_stroke_intersection(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_new = df.copy()
    scaler = MinMaxScaler()
    df_new[numeric_cols] = scaler.fit_transform(df_new[numeric_cols])
    return df_new, scaler

def preprocess_statlog_heart_stroke_union(df: pd.DataFrame):
    df_new = df.copy()
    drop_cols = ["cp_type", "sbp", "chol", "fbs", "ecg", "hr", "angina", "oldpeak", "slope", "mv", "thal"]
    df_new = df_new.drop(columns=drop_cols)
    mode_cols = ["hypertension_status", "heart_disease_status", "marital_status", "work_type", "residence_type", "smoking_status"]
    df_new[mode_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df_new[mode_cols])
    mean_cols = ["glucose_level", "bmi"]
    df_new[mean_cols] = SimpleImputer(strategy="mean").fit_transform(df_new[mean_cols])
    cols = df_new.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    df_new[cols] = scaler.fit_transform(df_new[cols])
    return df_new, scaler

def preprocess_heart_stroke_intersection(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_new = df.copy()
    scaler = MinMaxScaler()
    df_new[numeric_cols] = scaler.fit_transform(df_new[numeric_cols])
    return df_new, scaler

def preprocess_heart_stroke_union(df: pd.DataFrame):
    df_new = df.copy()
    mode_cols = ["cp_type", "fbs", "ecg", "angina", "slope", "thal", "hypertension_status", "heart_disease_status", "marital_status", "work_type", "residence_type", "smoking_status"]
    df_new[mode_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df_new[mode_cols])
    mean_cols = ["sbp", "chol", "hr", "oldpeak", "mv", "glucose_level", "bmi"]
    df_new[mean_cols] = SimpleImputer(strategy="mean").fit_transform(df_new[mean_cols])
    cols = df_new.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    df_new[cols] = scaler.fit_transform(df_new[cols])
    return df_new, scaler
# =====

# =====
# trustmebro
def preprocess_framingham_heart_union(df: pd.DataFrame):
    df_processed = df.copy()
    known_cats = ['edu', 'smoking_status', 'bp_status', 'stroke_status', 
                  'hypertension_status', 'diabetes_status', 'sex']
    categorical_cols = [col for col in known_cats if col in df_processed.columns]
    numeric_cols = [col for col in df_processed.columns 
                    if col not in categorical_cols and col != 'target']
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df_processed[numeric_cols])
    if len(categorical_cols) > 0:
        df_processed[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df_processed[categorical_cols])
    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    return df_processed, scaler

def preprocess_framingham_heart_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    return df_processed, scaler

def preprocess_chd_stroke_union(df: pd.DataFrame):
    df_processed = df.copy()
    known_cats = ['sex', 'hypertension_status', 'heart_disease_status', 
                  'marital_status', 'work_type', 'residence_type', 'smoking_status']
    categorical_cols = [col for col in known_cats if col in df_processed.columns]
    numeric_cols = [col for col in df_processed.columns 
                    if col not in categorical_cols and col != 'target']
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df_processed[numeric_cols])
    if len(categorical_cols) > 0:
        df_processed[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df_processed[categorical_cols])
    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    return df_processed, scaler

def preprocess_chd_stroke_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    return df_processed, scaler

def preprocess_chd_heart_stroke_union(df: pd.DataFrame):
    df_processed = df.copy()
    known_cats = ['sex', 'hypertension_status', 'heart_disease_status', 
                  'marital_status', 'work_type', 'residence_type', 'smoking_status']
    categorical_cols = [col for col in known_cats if col in df_processed.columns]
    numeric_cols = [col for col in df_processed.columns 
                    if col not in categorical_cols and col != 'target']
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df_processed[numeric_cols])
    if len(categorical_cols) > 0:
        df_processed[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df_processed[categorical_cols])
    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    return df_processed, scaler

def preprocess_chd_heart_stroke_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    return df_processed, scaler

def preprocess_framingham_heart_stroke_union(df: pd.DataFrame):
    df_processed = df.copy()
    known_cats = ['edu', 'smoking_status', 'bp_status', 'stroke_status',
                  'hypertension_status', 'diabetes_status', 
                  'heart_disease_status', 'marital_status', 
                  'work_type', 'residence_type', 'sex']
    categorical_cols = [col for col in known_cats if col in df_processed.columns]
    numeric_cols = [col for col in df_processed.columns 
                    if col not in categorical_cols and col != 'target']
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df_processed[numeric_cols])
    if len(categorical_cols) > 0:
        df_processed[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df_processed[categorical_cols])
    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    return df_processed, scaler

def preprocess_framingham_heart_stroke_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    return df_processed, scaler
# =====

# =====
# muffinhead
def preprocess_statlog_chd_stroke_union(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    categorical_cols = df_processed.select_dtypes(exclude=['number']).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_statlog_chd_stroke_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_statlog_chd_framingham_union(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    categorical_cols = df_processed.select_dtypes(exclude=['number']).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_statlog_chd_framingham_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_statlog_chd_union(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    categorical_cols = df_processed.select_dtypes(exclude=['number']).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_statlog_chd_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_statlog_framingham_union(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    categorical_cols = df_processed.select_dtypes(exclude=['number']).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_statlog_framingham_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler
# =====

# =====
# friend
def preprocess_statlog_chd_heart_intersection(df: pd.DataFrame):
    list_cat = []
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        list_cat.append(i)
    for i in ["famhist", "target"]:
        if i not in list_cat:
            list_cat.append(i)
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        if i not in list_cat:
            list_cat.append(i)

    df_processed = df.copy()
    for col in df_processed.columns:
        if col in list_cat:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    scaler = MinMaxScaler()
    scaled = df_processed.copy()
    numeric_to_scale = [col for col in scaled.columns if col not in list_cat]
    scaled[numeric_to_scale] = scaler.fit_transform(scaled[numeric_to_scale])

    return scaled, scaler

def preprocess_statlog_chd_heart_union(df: pd.DataFrame):
    list_cat = []
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        list_cat.append(i)
    for i in ["famhist", "target"]:
        if i not in list_cat:
            list_cat.append(i)
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        if i not in list_cat:
            list_cat.append(i)

    df_processed = df.copy()
    for col in df_processed.columns:
        if col in list_cat:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    scaler = MinMaxScaler()
    scaled = df_processed.copy()
    numeric_to_scale = [col for col in scaled.columns if col not in list_cat]
    scaled[numeric_to_scale] = scaler.fit_transform(scaled[numeric_to_scale])

    return scaled, scaler

def preprocess_statlog_framingham_heart_intersection(df: pd.DataFrame):
    list_cat = []
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        list_cat.append(i)
    for i in ["sex", "edu", "smoking_status", "bp_status", "stroke_status", "hypertension_status", "diabetes_status", "target"]:
        if i not in list_cat:
            list_cat.append(i)
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        if i not in list_cat:
            list_cat.append(i)

    df_processed = df.copy()
    for col in df_processed.columns:
        if col in list_cat:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    scaler = MinMaxScaler()
    scaled = df_processed.copy()
    numeric_to_scale = [col for col in scaled.columns if col not in list_cat]
    scaled[numeric_to_scale] = scaler.fit_transform(scaled[numeric_to_scale])

    return scaled, scaler

def preprocess_statlog_framingham_heart_union(df: pd.DataFrame):
    list_cat = []
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        list_cat.append(i)
    for i in ["sex", "edu", "smoking_status", "bp_status", "stroke_status", "hypertension_status", "diabetes_status", "target"]:
        if i not in list_cat:
            list_cat.append(i)
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        if i not in list_cat:
            list_cat.append(i)

    df_processed = df.copy()
    for col in df_processed.columns:
        if col in list_cat:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    scaler = MinMaxScaler()
    scaled = df_processed.copy()
    numeric_to_scale = [col for col in scaled.columns if col not in list_cat]
    scaled[numeric_to_scale] = scaler.fit_transform(scaled[numeric_to_scale])

    return scaled, scaler

def preprocess_statlog_heart_intersection(df: pd.DataFrame):
    list_cat = []
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        list_cat.append(i)
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        if i not in list_cat:
            list_cat.append(i)

    df_processed = df.copy()
    for col in df_processed.columns:
        if col in list_cat:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    scaler = MinMaxScaler()
    scaled = df_processed.copy()
    numeric_to_scale = [col for col in scaled.columns if col not in list_cat]
    scaled[numeric_to_scale] = scaler.fit_transform(scaled[numeric_to_scale])

    return scaled, scaler

def preprocess_statlog_heart_union(df: pd.DataFrame):
    list_cat = []
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        list_cat.append(i)
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        if i not in list_cat:
            list_cat.append(i)

    df_processed = df.copy()
    for col in df_processed.columns:
        if col in list_cat:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    scaler = MinMaxScaler()
    scaled = df_processed.copy()
    numeric_to_scale = [col for col in scaled.columns if col not in list_cat]
    scaled[numeric_to_scale] = scaler.fit_transform(scaled[numeric_to_scale])

    return scaled, scaler

def preprocess_statlog_stroke_intersection(df: pd.DataFrame):
    list_cat = []
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        list_cat.append(i)
    for i in ["sex", "hypertension_status", "heart_disease_status", "marital_status", "work_type", "residence_type", "smoking_status", "target"]:
        if i not in list_cat:
            list_cat.append(i)

    df_processed = df.copy()
    for col in df_processed.columns:
        if col in list_cat:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    scaler = MinMaxScaler()
    scaled = df_processed.copy()
    numeric_to_scale = [col for col in scaled.columns if col not in list_cat]
    scaled[numeric_to_scale] = scaler.fit_transform(scaled[numeric_to_scale])

    return scaled, scaler

def preprocess_statlog_stroke_union(df: pd.DataFrame):
    list_cat = []
    for i in ["sex", "cp_type", "fbs", "ecg", "angina", "slope", "thal", "target"]:
        list_cat.append(i)
    for i in ["sex", "hypertension_status", "heart_disease_status", "marital_status", "work_type", "residence_type", "smoking_status", "target"]:
        if i not in list_cat:
            list_cat.append(i)

    df_processed = df.copy()
    for col in df_processed.columns:
        if col in list_cat:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    scaler = MinMaxScaler()
    scaled = df_processed.copy()
    numeric_to_scale = [col for col in scaled.columns if col not in list_cat]
    scaled[numeric_to_scale] = scaler.fit_transform(scaled[numeric_to_scale])

    return scaled, scaler
# =====

# =====
# wsuperwhalew
def preprocess_chd_framingham_heart_union(df: pd.DataFrame):
    df_processed = df.copy()

    missing_fractions = df_processed.isnull().mean()
    cols_to_drop = missing_fractions[missing_fractions > 0.75].index
    if len(cols_to_drop) > 0:
        df_processed.drop(columns=cols_to_drop, inplace=True)

    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    categorical_cols = df_processed.select_dtypes(exclude=['number']).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_chd_framingham_heart_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_chd_framingham_stroke_union(df: pd.DataFrame):
    df_processed = df.copy()

    missing_fractions = df_processed.isnull().mean()
    cols_to_drop = missing_fractions[missing_fractions > 0.75].index
    if len(cols_to_drop) > 0:
        df_processed.drop(columns=cols_to_drop, inplace=True)

    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    categorical_cols = df_processed.select_dtypes(exclude=['number']).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_chd_framingham_stroke_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_chd_framingham_union(df: pd.DataFrame):
    df_processed = df.copy()

    missing_fractions = df_processed.isnull().mean()
    cols_to_drop = missing_fractions[missing_fractions > 0.75].index
    if len(cols_to_drop) > 0:
        df_processed.drop(columns=cols_to_drop, inplace=True)

    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    categorical_cols = df_processed.select_dtypes(exclude=['number']).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_chd_framingham_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler

def preprocess_chd_heart_union(df: pd.DataFrame):
    df_processed = df.copy()

    missing_fractions = df_processed.isnull().mean()
    cols_to_drop = missing_fractions[missing_fractions > 0.75].index
    if len(cols_to_drop) > 0:
        df_processed.drop(columns=cols_to_drop, inplace=True)

    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    categorical_cols = df_processed.select_dtypes(exclude=['number']).columns

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler


def preprocess_chd_heart_intersection(df: pd.DataFrame):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns

    scaler = MinMaxScaler()
    if len(numeric_cols) > 0:
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    return df_processed, scaler
# =====

def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: list, ncols: int = 4):
    nrows = -(-len(categorical_cols) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 4))
    axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        sns.countplot(y=col, data=df, ax=axes[i])
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel('')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle(f'Categorical Distributions — {df.attrs.get("dataset_name", "Unknown")} Dataset', y=1.01)
    plt.tight_layout()
    plt.show()
        
def plot_numerical_heatmap(df: pd.DataFrame, numerical_cols: list):
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
    plt.title(f'Correlation Heatmap of Numerical Features in {df.attrs.get("dataset_name", "Unknown")} Dataset')
    plt.show()
    
def set_unscaled(df: pd.DataFrame, dataset_name: str):
    df.to_csv(f'data/unscaled/{dataset_name}.csv', index=False)
    
def set_test(df: pd.DataFrame, dataset_name: str):
    df.to_csv(f'data/test/{dataset_name}.csv', index=False)
    
def set_unscaled_combined(df: pd.DataFrame):
    dataset_name = df.attrs.get('dataset_name', 'Combined').lower()
    df.to_csv(f'data/unscaled/combined/{dataset_name}.csv', index=False)
    
def set_preprocessed(df: pd.DataFrame, dataset_name: str, scaler: MinMaxScaler, scaler_name: str):
    if dataset_name is None:
        dataset_name = df.attrs.get('dataset_name', 'Unknown').lower()
    
    if scaler_name is None:
        scaler_name = dataset_name
    
    df.to_csv(f'data/processed/{dataset_name}.csv', index=False)
    scaler_file = f'scaler/{scaler_name}.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
def set_preprocessed_combined(df: pd.DataFrame, dataset_name: str, scaler: MinMaxScaler, scaler_name: str):
    if dataset_name is None:
        dataset_name = df.attrs.get('dataset_name', 'Unknown').lower()
    
    if scaler_name is None:
        scaler_name = dataset_name
    
    df.to_csv(f'data/processed/combined/{dataset_name}.csv', index=False)
    scaler_file = f'scaler/combined/{scaler_name}.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
        
def get_unscaled(dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(f'data/unscaled/{dataset_name}.csv')
    df.attrs['dataset_name'] = dataset_name
    return df

def get_unscaled_combined(dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(f'data/unscaled/combined/{dataset_name}.csv')
    df.attrs['dataset_name'] = dataset_name
    return df

def get_preprocessed(dataset_name: str, scaler_name: str) -> tuple:
    df = pd.read_csv(f'data/processed/{dataset_name}.csv')
    df.attrs['dataset_name'] = dataset_name
    with open(f'scaler/{scaler_name}.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return df, scaler

def get_test(dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(f'data/test/{dataset_name}.csv')
    df.attrs['dataset_name'] = dataset_name
    return df

def get_preprocessed_combined(dataset_name: str, scaler_name: str) -> pd.DataFrame:
    df = pd.read_csv(f'data/processed/combined/{dataset_name}.csv')
    df.attrs['dataset_name'] = dataset_name
    with open(f'scaler/combined/{scaler_name}.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return df, scaler

def combine_datasets_union(df1: pd.DataFrame, df2: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.attrs['dataset_name'] = dataset_name
    return combined_df

def combine_datasets_intersection(df1: pd.DataFrame, df2: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    common_cols = df1.columns.intersection(df2.columns)
    combined_df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.attrs['dataset_name'] = dataset_name
    return combined_df

# ===========================================================================
# TRAIN

import os
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

N_SPLITS = 5
MODEL_DIR = "model"

def split_xy(df: pd.DataFrame, target_col: str = "target"):
    if target_col is None or target_col not in df.columns:
        raise ValueError("Cannot find target column")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def evaluate_and_fit(name: str, estimator, X, y, cv: StratifiedKFold, scoring: dict = None):
    cv_res = cross_validate(
        estimator, X, y,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {k: (np.mean(v), np.std(v)) for k, v in cv_res.items() if k.startswith("test_")}
    # for metric, (m, s) in summary.items():
    #     print(f"{name:>12} - {metric.replace('test_', ''):>9}: {m:.4f} ± {s:.4f}")

    estimator.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(estimator, model_path)

    return summary

def train_dataset(dataset_name: str, df: pd.DataFrame, scaler: MinMaxScaler, cv: StratifiedKFold = None, scoring: dict = None):
    X, y = split_xy(df)

    scaler_path = os.path.join(MODEL_DIR, f"{dataset_name}_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    ratio = (y == 0).sum() / (y == 1).sum()
    is_imbalanced = ratio >= 2.0

    def make_models(imbalanced: bool):
        def wrap(clf):
            if imbalanced:
                return ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", clf)])
            return clf

        return {
            "logreg": wrap(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            "knn": wrap(KNeighborsClassifier(n_neighbors=5)),
            "decisiontree": wrap(DecisionTreeClassifier(random_state=RANDOM_STATE)),
            "svm": wrap(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            "naivebayes": wrap(GaussianNB()),
            "randomforest": wrap(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
            "xgb": wrap(XGBClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss", verbosity=0)),
            "lightgbm": wrap(LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)),
        }

    models = make_models(is_imbalanced)

    results = {}
    for model_name, est in models.items():
        print(f"Training {dataset_name} dataset with {model_name}.")
        full_name = f"{dataset_name}_{model_name}"
        results[full_name] = evaluate_and_fit(full_name, est, X, y, cv, scoring)

    return results

def show_result(model_result_set: set):
    return pd.DataFrame({
        "dataset": ["_".join(k.split("_")[:-1]) for k in model_result_set.keys()],
        "model": [k.split("_")[-1] for k in model_result_set.keys()],
        'accuracy_mean': [v['test_accuracy'][0] for v in model_result_set.values()],
        # 'precision_mean': [v['test_precision'][0] for v in model_result_set.values()],
        # 'recall_mean': [v['test_recall'][0] for v in model_result_set.values()],
        "f1_mean": [v['test_f1'][0] for v in model_result_set.values()],
        "f1_macro_mean": [v['test_f1_macro'][0] for v in model_result_set.values()],
        "recall_minority_mean": [v['test_recall_minority'][0] for v in model_result_set.values()],
    })
    
def select_best_model(dataset_name: str, result_dataset: pd.DataFrame):
    filtered_dataset = result_dataset[result_dataset["dataset"].str.contains(dataset_name)]
    max_f1_value = filtered_dataset["f1_mean"].max()
    selected_row = filtered_dataset[filtered_dataset["f1_mean"] == max_f1_value].iloc[0, :2]
    selected_model_name = selected_row["dataset"] + "_" + selected_row["model"]
    selected_model = joblib.load(f'{MODEL_DIR}/{selected_model_name}.joblib')
    return selected_model, max_f1_value