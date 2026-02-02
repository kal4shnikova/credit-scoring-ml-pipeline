"""
Модуль для выполнения предсказаний с помощью обученной модели
"""
import pandas as pd
import joblib
from pathlib import Path
import numpy as np


def load_model(model_path: str):
    """
    Загрузка обученной модели
    
    Args:
        model_path: путь к файлу модели
        
    Returns:
        Загруженная модель
    """
    print(f"📥 Загрузка модели из {model_path}...")
    model = joblib.load(model_path)
    print("✅ Модель загружена успешно")
    return model


def predict(model, X: pd.DataFrame) -> tuple:
    """
    Выполнение предсказаний
    
    Args:
        model: обученная модель
        X: DataFrame с признаками
        
    Returns:
        Кортеж (predictions, probabilities)
    """
    print(f"🎯 Выполнение предсказаний для {len(X)} записей...")
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    print("✅ Предсказания выполнены")
    
    return predictions, probabilities


def predict_single(model, features: dict) -> dict:
    """
    Предсказание для одного клиента
    
    Args:
        model: обученная модель
        features: словарь с признаками клиента
        
    Returns:
        Словарь с результатом предсказания
    """
    # Создание DataFrame из словаря
    df = pd.DataFrame([features])
    
    # Предсказание
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0, 1]
    
    result = {
        'default_prediction': int(prediction),
        'default_probability': float(probability),
        'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
    }
    
    return result


def main():
    """
    Пример использования модуля для предсказаний
    """
    # Определение путей
    project_dir = Path(__file__).resolve().parents[2]
    models_dir = project_dir / 'models'
    model_path = models_dir / 'credit_default_model_gradient_boosting.pkl'
    
    # Проверка наличия модели
    if not model_path.exists():
        print(f"❌ Модель не найдена: {model_path}")
        print("Сначала выполните обучение модели: python src/models/train.py")
        return
    
    # Загрузка модели
    model = load_model(model_path)
    
    # Пример данных клиента
    sample_client = {
        'LIMIT_BAL': 20000.0,
        'SEX': 2,
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 24,
        'PAY_0': 2,
        'PAY_2': 2,
        'PAY_3': -1,
        'PAY_4': -1,
        'PAY_5': -2,
        'PAY_6': -2,
        'BILL_AMT1': 3913.0,
        'BILL_AMT2': 3102.0,
        'BILL_AMT3': 689.0,
        'BILL_AMT4': 0.0,
        'BILL_AMT5': 0.0,
        'BILL_AMT6': 0.0,
        'PAY_AMT1': 0.0,
        'PAY_AMT2': 689.0,
        'PAY_AMT3': 0.0,
        'PAY_AMT4': 0.0,
        'PAY_AMT5': 0.0,
        'PAY_AMT6': 0.0
    }
    
    # Предсказание для одного клиента
    print("\n" + "="*50)
    print("🔮 Пример предсказания для клиента:")
    print("="*50)
    result = predict_single(model, sample_client)
    for key, value in result.items():
        print(f"   {key}: {value}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
