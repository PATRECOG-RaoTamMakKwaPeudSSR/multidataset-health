import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle

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

def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: list, ncols: int = 4):
    nrows = -(-len(categorical_cols) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 4))
    axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        sns.countplot(x=col, data=df, ax=axes[i])
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
    
def set_unscaled_combined(df: pd.DataFrame):
    dataset_name = df.attrs.get('dataset_name', 'Combined').lower()
    df.to_csv(f'data/unscaled/combined/{dataset_name}.csv', index=False)
    
def set_preprocessed(df: pd.DataFrame, dataset_name: str, scaler: MinMaxScaler, scaler_name: str):
    df.to_csv(f'data/processed/{dataset_name}.csv', index=False)
    scaler_file = f'scaler/{scaler_name}.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
def set_preprocessed_combined(df: pd.DataFrame, scaler: MinMaxScaler, scaler_name: str):
    dataset_name = df.attrs.get('dataset_name', 'Combined').lower()
    df.to_csv(f'data/processed/combined/{dataset_name}.csv', index=False)
    scaler_file = f'scaler/combined/{scaler_name}.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
        
def get_unscaled(dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(f'data/unscaled/{dataset_name}.csv')
    return df

def get_unscaled_combined(dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(f'data/unscaled/combined/{dataset_name}.csv')
    return df

def get_preprocessed(dataset_name: str, scaler_name: str) -> tuple:
    df = pd.read_csv(f'data/processed/{dataset_name}.csv')
    with open(f'scaler/{scaler_name}.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return df, scaler

def get_preprocessed_combined(dataset_name: str, scaler_name: str) -> pd.DataFrame:
    df = pd.read_csv(f'data/processed/combined/{dataset_name}.csv')
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