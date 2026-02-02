"""
Модуль для создания sklearn Pipeline
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


def create_pipeline(
    numeric_features: list,
    categorical_features: list,
    model_type: str = "gradient_boosting",
    **model_params
) -> Pipeline:
    """
    Создание sklearn Pipeline с предобработкой и моделью
    
    Args:
        numeric_features: список числовых признаков
        categorical_features: список категориальных признаков
        model_type: тип модели ("gradient_boosting" или "logistic_regression")
        **model_params: параметры для модели
        
    Returns:
        Pipeline объект
    """
    
    # Трансформер для числовых признаков
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Трансформер для категориальных признаков
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Объединение трансформеров
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Выбор модели
    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            random_state=42,
            **model_params
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            **model_params
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    # Создание полного пайплайна
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline


def get_feature_lists():
    """
    Получение списков числовых и категориальных признаков
    
    Returns:
        Кортеж (numeric_features, categorical_features)
    """
    
    # Числовые признаки
    numeric_features = [
        'LIMIT_BAL', 'AGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
        # Созданные признаки
        'avg_payment_delay', 'max_payment_delay', 'num_months_delayed',
        'payment_trend', 'avg_bill_amt', 'max_bill_amt', 'std_bill_amt',
        'bill_trend', 'bill_amt_ratio', 'avg_payment_amt', 'max_payment_amt',
        'num_zero_payments', 'payment_amt_trend', 'credit_utilization',
        'avg_credit_utilization', 'payment_to_bill_ratio',
        'avg_payment_to_bill_ratio', 'limit_per_age'
    ]
    
    # Категориальные признаки
    categorical_features = [
        'SEX', 'EDUCATION', 'MARRIAGE', 'has_payment_delay', 'age_group'
    ]
    
    return numeric_features, categorical_features
