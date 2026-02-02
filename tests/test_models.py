"""
Unit-тесты для модуля models
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from models.pipeline import create_pipeline, get_feature_lists
from sklearn.pipeline import Pipeline


class TestModelsModule:
    """Тесты для модуля моделей"""
    
    def test_get_feature_lists(self):
        """Тест получения списков признаков"""
        numeric_features, categorical_features = get_feature_lists()
        
        # Проверяем, что списки не пустые
        assert len(numeric_features) > 0
        assert len(categorical_features) > 0
        
        # Проверяем, что базовые признаки присутствуют
        assert 'LIMIT_BAL' in numeric_features
        assert 'AGE' in numeric_features
        assert 'SEX' in categorical_features
        assert 'EDUCATION' in categorical_features
    
    def test_create_pipeline_gradient_boosting(self):
        """Тест создания pipeline с GradientBoosting"""
        numeric_features, categorical_features = get_feature_lists()
        
        pipeline = create_pipeline(
            numeric_features,
            categorical_features,
            model_type="gradient_boosting",
            n_estimators=10
        )
        
        # Проверяем, что это Pipeline
        assert isinstance(pipeline, Pipeline)
        
        # Проверяем наличие основных шагов
        assert 'preprocessor' in pipeline.named_steps
        assert 'classifier' in pipeline.named_steps
    
    def test_create_pipeline_logistic_regression(self):
        """Тест создания pipeline с LogisticRegression"""
        numeric_features, categorical_features = get_feature_lists()
        
        pipeline = create_pipeline(
            numeric_features,
            categorical_features,
            model_type="logistic_regression",
            C=1.0
        )
        
        # Проверяем, что это Pipeline
        assert isinstance(pipeline, Pipeline)
        assert 'preprocessor' in pipeline.named_steps
        assert 'classifier' in pipeline.named_steps
    
    def test_create_pipeline_invalid_model_type(self):
        """Тест что недопустимый тип модели вызывает ошибку"""
        numeric_features, categorical_features = get_feature_lists()
        
        with pytest.raises(ValueError):
            create_pipeline(
                numeric_features,
                categorical_features,
                model_type="invalid_model"
            )
    
    def test_pipeline_fit_predict(self):
        """Тест что pipeline может обучаться и делать предсказания"""
        # Создаем простой тестовый датасет
        np.random.seed(42)
        n_samples = 100
        
        # Создаем минимальный набор признаков
        X = pd.DataFrame({
            'LIMIT_BAL': np.random.uniform(10000, 100000, n_samples),
            'AGE': np.random.randint(20, 70, n_samples),
            'SEX': np.random.choice([1, 2], n_samples),
            'EDUCATION': np.random.choice([1, 2, 3, 4], n_samples),
            'MARRIAGE': np.random.choice([1, 2, 3], n_samples),
            'PAY_0': np.random.randint(-2, 3, n_samples),
            'PAY_2': np.random.randint(-2, 3, n_samples),
            'PAY_3': np.random.randint(-2, 3, n_samples),
            'PAY_4': np.random.randint(-2, 3, n_samples),
            'PAY_5': np.random.randint(-2, 3, n_samples),
            'PAY_6': np.random.randint(-2, 3, n_samples),
            'BILL_AMT1': np.random.uniform(0, 50000, n_samples),
            'BILL_AMT2': np.random.uniform(0, 50000, n_samples),
            'BILL_AMT3': np.random.uniform(0, 50000, n_samples),
            'BILL_AMT4': np.random.uniform(0, 50000, n_samples),
            'BILL_AMT5': np.random.uniform(0, 50000, n_samples),
            'BILL_AMT6': np.random.uniform(0, 50000, n_samples),
            'PAY_AMT1': np.random.uniform(0, 10000, n_samples),
            'PAY_AMT2': np.random.uniform(0, 10000, n_samples),
            'PAY_AMT3': np.random.uniform(0, 10000, n_samples),
            'PAY_AMT4': np.random.uniform(0, 10000, n_samples),
            'PAY_AMT5': np.random.uniform(0, 10000, n_samples),
            'PAY_AMT6': np.random.uniform(0, 10000, n_samples),
            # Добавляем feature engineering признаки
            'avg_payment_delay': np.random.uniform(-1, 2, n_samples),
            'max_payment_delay': np.random.randint(0, 3, n_samples),
            'num_months_delayed': np.random.randint(0, 6, n_samples),
            'has_payment_delay': np.random.choice([0, 1], n_samples),
            'payment_trend': np.random.uniform(-3, 3, n_samples),
            'avg_bill_amt': np.random.uniform(0, 40000, n_samples),
            'max_bill_amt': np.random.uniform(0, 50000, n_samples),
            'std_bill_amt': np.random.uniform(0, 10000, n_samples),
            'bill_trend': np.random.uniform(-10000, 10000, n_samples),
            'bill_amt_ratio': np.random.uniform(0, 2, n_samples),
            'avg_payment_amt': np.random.uniform(0, 8000, n_samples),
            'max_payment_amt': np.random.uniform(0, 10000, n_samples),
            'num_zero_payments': np.random.randint(0, 6, n_samples),
            'payment_amt_trend': np.random.uniform(-5000, 5000, n_samples),
            'credit_utilization': np.random.uniform(0, 1, n_samples),
            'avg_credit_utilization': np.random.uniform(0, 1, n_samples),
            'payment_to_bill_ratio': np.random.uniform(0, 2, n_samples),
            'avg_payment_to_bill_ratio': np.random.uniform(0, 2, n_samples),
            'limit_per_age': np.random.uniform(100, 3000, n_samples),
            'age_group': np.random.choice(['young', 'adult', 'middle', 'senior', 'elderly'], n_samples)
        })
        
        y = np.random.choice([0, 1], n_samples)
        
        # Создаем pipeline
        numeric_features, categorical_features = get_feature_lists()
        pipeline = create_pipeline(
            numeric_features,
            categorical_features,
            model_type="logistic_regression",
            C=1.0
        )
        
        # Обучаем
        pipeline.fit(X, y)
        
        # Проверяем предсказания
        predictions = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)
        
        assert len(predictions) == n_samples
        assert probabilities.shape == (n_samples, 2)
        assert np.all((predictions == 0) | (predictions == 1))
