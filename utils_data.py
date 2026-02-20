# utils_data.py
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "archive", "Training_disease_prediction.csv")
TEST_CSV  = os.path.join(BASE_DIR, "archive", "Testing_disease_prediction.csv")

def load_and_clean():
    """
    Loads training and testing CSVs, drops stray 'Unnamed: 133' if present,
    aligns test columns with train columns.
    Returns (train_df, test_df).
    """
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # drop stray unnamed col if present
    if "Unnamed: 133" in train.columns:
        train = train.drop("Unnamed: 133", axis=1)

    # ensure test has same feature columns as train (fill missing with 0)
    X_train = train.drop(columns=["prognosis"])
    X_test = test.drop(columns=["prognosis"])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # put prognosis column back into X_test aligned
    test_aligned = X_test.copy()
    test_aligned["prognosis"] = test["prognosis"].values

    return train, test_aligned

def get_symptom_list(train_df):
    return train_df.drop(columns=["prognosis"]).columns.tolist()

def generate_doctor_mapping(diseases):
    """
    Heuristic mapper: returns dict {disease_name: suggested_specialist}
    Use this as a seed and refine manually if needed.
    """
    mapping = {}
    for d in diseases:
        ld = d.lower()
        if any(k in ld for k in ["psoriasis", "eczema", "rash", "dermat", "fungal", "skin"]):
            mapping[d] = "Dermatologist"
        elif any(k in ld for k in ["heart", "cardio", "hypertension", "chest pain", "heart attack"]):
            mapping[d] = "Cardiologist"
        elif any(k in ld for k in ["diabetes", "insulin", "endocrin"]):
            mapping[d] = "Endocrinologist"
        elif any(k in ld for k in ["asthma", "pneumonia", "tuberculosis", "bronch", "copd"]):
            mapping[d] = "Pulmonologist"
        elif any(k in ld for k in ["arthritis", "joint", "bone", "back pain"]):
            mapping[d] = "Orthopedic"
        elif any(k in ld for k in ["migraine", "headache", "seizure", "neu"]):
            mapping[d] = "Neurologist"
        elif any(k in ld for k in ["eye", "conjunct", "vision"]):
            mapping[d] = "Ophthalmologist"
        elif any(k in ld for k in ["preg", "obstet", "gyn", "uterus"]):
            mapping[d] = "Gynecologist"
        elif any(k in ld for k in ["stomach", "ulcer", "hepat", "liver", "gastr", "abdomen"]):
            mapping[d] = "Gastroenterologist"
        elif any(k in ld for k in ["urine", "urinary", "kidney"]):
            mapping[d] = "Nephrologist"
        else:
            mapping[d] = "General Physician"
    return mapping
