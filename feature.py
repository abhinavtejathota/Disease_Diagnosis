import numpy as np
import pandas as pd

def get_features_for_heart(data: dict):
    feature_order = [
        'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    return pd.DataFrame([data], columns=feature_order)

def get_features_for_hepatitis(data: dict):
    feature_order = [
        'Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL',
        'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'
    ]
    return pd.DataFrame([data], columns=feature_order)

def get_features_for_stroke(data: dict):
    feature_order = [
        'age', 'hypertension', 'heart_disease', 'ever_married',
        'avg_glucose_level', 'bmi', 'gender_Male', 'gender_Other',
        'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
        'work_type_Self-employed', 'work_type_children',
        'Residence_type_Rural', 'Residence_type_Urban',
        'smoking_status_Unknown', 'smoking_status_formerly smoked',
        'smoking_status_never smoked', 'smoking_status_smokes'
    ]
    return pd.DataFrame([data], columns=feature_order)

def prepare_features(disease_type: str, input_data: dict):
    if disease_type.lower() == "heart":
        return get_features_for_heart(input_data)
    elif disease_type.lower() == "hepatitis":
        return get_features_for_hepatitis(input_data)
    elif disease_type.lower() == "stroke":
        return get_features_for_stroke(input_data)
    else:
        raise ValueError("Invalid disease type. Choose from 'heart', 'hepatitis', 'stroke'.")
