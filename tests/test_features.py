"""
Unit-тесты для модуля features
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from features.build_features import (
    create_payment_features,
    create_bill_features,
    create_payment_amt_features,
    create_utilization_features
)


class TestFeaturesModule:
    """Тесты для модуля feature engineering"""
    
    def test_create_payment_features(self):
        """Тест создания признаков истории платежей"""
        df = pd.DataFrame({
            'PAY_0': [2, -1, 0],
            'PAY_2': [2, 0, 1],
            'PAY_3': [-1, -1, 2],
            'PAY_4': [-1, -2, 0],
            'PAY_5': [-2, 0, 1],
            'PAY_6': [-2, 1, 0]
        })
        
        df_features = create_payment_features(df)
        
        # Проверяем, что новые признаки созданы
        assert 'avg_payment_delay' in df_features.columns
        assert 'max_payment_delay' in df_features.columns
        assert 'num_months_delayed' in df_features.columns
        assert 'has_payment_delay' in df_features.columns
        assert 'payment_trend' in df_features.columns
        
        # Проверяем корректность вычислений
        assert df_features['has_payment_delay'].iloc[0] == 1  # есть задержки
        assert df_features['has_payment_delay'].iloc[1] == 1  # есть задержки
        assert df_features['max_payment_delay'].iloc[0] == 2
    
    def test_create_bill_features(self):
        """Тест создания признаков счетов"""
        df = pd.DataFrame({
            'BILL_AMT1': [1000, 2000, 3000],
            'BILL_AMT2': [900, 1800, 2700],
            'BILL_AMT3': [800, 1600, 2400],
            'BILL_AMT4': [700, 1400, 2100],
            'BILL_AMT5': [600, 1200, 1800],
            'BILL_AMT6': [500, 1000, 1500]
        })
        
        df_features = create_bill_features(df)
        
        # Проверяем наличие новых признаков
        assert 'avg_bill_amt' in df_features.columns
        assert 'max_bill_amt' in df_features.columns
        assert 'std_bill_amt' in df_features.columns
        assert 'bill_trend' in df_features.columns
        assert 'bill_amt_ratio' in df_features.columns
        
        # Проверяем значения
        assert df_features['max_bill_amt'].iloc[0] == 1000
        assert df_features['bill_trend'].iloc[0] == 500  # 1000 - 500
    
    def test_create_payment_amt_features(self):
        """Тест создания признаков сумм платежей"""
        df = pd.DataFrame({
            'PAY_AMT1': [1000, 0, 500],
            'PAY_AMT2': [900, 0, 400],
            'PAY_AMT3': [800, 0, 300],
            'PAY_AMT4': [700, 100, 200],
            'PAY_AMT5': [600, 0, 100],
            'PAY_AMT6': [500, 0, 50]
        })
        
        df_features = create_payment_amt_features(df)
        
        # Проверяем наличие признаков
        assert 'avg_payment_amt' in df_features.columns
        assert 'max_payment_amt' in df_features.columns
        assert 'num_zero_payments' in df_features.columns
        assert 'payment_amt_trend' in df_features.columns
        
        # Проверяем подсчет нулевых платежей
        assert df_features['num_zero_payments'].iloc[1] == 5  # 5 нулевых платежей
    
    def test_create_utilization_features(self):
        """Тест создания признаков использования кредита"""
        df = pd.DataFrame({
            'LIMIT_BAL': [10000, 20000, 30000],
            'BILL_AMT1': [5000, 10000, 15000],
            'BILL_AMT2': [4000, 8000, 12000],
            'BILL_AMT3': [3000, 6000, 9000],
            'BILL_AMT4': [2000, 4000, 6000],
            'BILL_AMT5': [1000, 2000, 3000],
            'BILL_AMT6': [500, 1000, 1500],
            'PAY_AMT1': [1000, 2000, 3000]
        })
        
        df_features = create_utilization_features(df)
        
        # Проверяем наличие признаков
        assert 'credit_utilization' in df_features.columns
        assert 'avg_credit_utilization' in df_features.columns
        assert 'payment_to_bill_ratio' in df_features.columns
        assert 'avg_payment_to_bill_ratio' in df_features.columns
        
        # Проверяем значения коэффициента использования
        # Для первой строки: 5000 / 10000 = 0.5
        expected_utilization = 5000 / 10000
        assert abs(df_features['credit_utilization'].iloc[0] - expected_utilization) < 0.01
    
    def test_features_do_not_modify_original(self):
        """Тест что функции не модифицируют исходный DataFrame"""
        df_original = pd.DataFrame({
            'PAY_0': [2, -1],
            'PAY_2': [2, 0],
            'PAY_3': [-1, -1],
            'PAY_4': [-1, -2],
            'PAY_5': [-2, 0],
            'PAY_6': [-2, 1]
        })
        
        original_columns = df_original.columns.tolist()
        
        df_features = create_payment_features(df_original)
        
        # Проверяем, что исходный DataFrame не изменился
        assert df_original.columns.tolist() == original_columns
        assert len(df_features.columns) > len(df_original.columns)
