"""
Unit-тесты для модуля data
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from data.make_dataset import clean_data, load_raw_data


class TestDataModule:
    """Тесты для модуля обработки данных"""
    
    def test_clean_data_removes_duplicates(self):
        """Тест удаления дубликатов"""
        # Создаем тестовый DataFrame с дубликатами
        df = pd.DataFrame({
            'ID': [1, 1, 2, 3],
            'LIMIT_BAL': [10000, 10000, 20000, 30000],
            'default.payment.next.month': [0, 0, 1, 0]
        })
        
        df_clean = clean_data(df)
        
        # Проверяем, что дубликаты удалены
        assert len(df_clean) == 3
        assert len(df_clean['ID'].unique()) == 3
    
    def test_clean_data_renames_target(self):
        """Тест переименования целевой переменной"""
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'default.payment.next.month': [0, 1, 0]
        })
        
        df_clean = clean_data(df)
        
        # Проверяем, что колонка переименована
        assert 'default' in df_clean.columns
        assert 'default.payment.next.month' not in df_clean.columns
    
    def test_clean_data_handles_education(self):
        """Тест корректировки значений EDUCATION"""
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'EDUCATION': [0, 1, 2, 5, 6],
            'default.payment.next.month': [0, 0, 1, 0, 1]
        })
        
        df_clean = clean_data(df)
        
        # Проверяем, что все значения в диапазоне 1-4
        assert df_clean['EDUCATION'].min() >= 1
        assert df_clean['EDUCATION'].max() <= 4
        # Проверяем, что 0, 5, 6 заменены на 4
        assert df_clean.loc[df_clean['ID'] == 1, 'EDUCATION'].values[0] == 4
        assert df_clean.loc[df_clean['ID'] == 4, 'EDUCATION'].values[0] == 4
        assert df_clean.loc[df_clean['ID'] == 5, 'EDUCATION'].values[0] == 4
    
    def test_clean_data_handles_marriage(self):
        """Тест корректировки значений MARRIAGE"""
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'MARRIAGE': [0, 1, 2],
            'default.payment.next.month': [0, 0, 1]
        })
        
        df_clean = clean_data(df)
        
        # Проверяем, что 0 заменено на 3
        assert df_clean.loc[df_clean['ID'] == 1, 'MARRIAGE'].values[0] == 3
    
    def test_clean_data_returns_dataframe(self):
        """Тест что функция возвращает DataFrame"""
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'default.payment.next.month': [0, 1, 0]
        })
        
        df_clean = clean_data(df)
        
        assert isinstance(df_clean, pd.DataFrame)
