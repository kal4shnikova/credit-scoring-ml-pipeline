"""
Модуль для Feature Engineering - создание новых признаков
"""
import pandas as pd
import numpy as np
from pathlib import Path


def create_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков на основе истории платежей
    
    Args:
        df: исходный DataFrame
        
    Returns:
        DataFrame с новыми признаками
    """
    df = df.copy()
    
    # Среднее значение задержки платежей (PAY_0 to PAY_6)
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    df['avg_payment_delay'] = df[pay_cols].mean(axis=1)
    
    # Максимальная задержка платежа
    df['max_payment_delay'] = df[pay_cols].max(axis=1)
    
    # Количество месяцев с задержкой платежа
    df['num_months_delayed'] = (df[pay_cols] > 0).sum(axis=1)
    
    # Есть ли задержки платежа вообще (бинарный признак)
    df['has_payment_delay'] = (df['num_months_delayed'] > 0).astype(int)
    
    # Тренд задержек (ухудшение/улучшение ситуации)
    # Положительное значение = ухудшение, отрицательное = улучшение
    df['payment_trend'] = df['PAY_0'] - df['PAY_6']
    
    return df


def create_bill_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков на основе счетов (BILL_AMT)
    
    Args:
        df: исходный DataFrame
        
    Returns:
        DataFrame с новыми признаками
    """
    df = df.copy()
    
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    
    # Средняя сумма счета
    df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
    
    # Максимальная сумма счета
    df['max_bill_amt'] = df[bill_cols].max(axis=1)
    
    # Стандартное отклонение суммы счета (волатильность)
    df['std_bill_amt'] = df[bill_cols].std(axis=1)
    
    # Тренд суммы счета (растет или падает)
    df['bill_trend'] = df['BILL_AMT1'] - df['BILL_AMT6']
    
    # Отношение последнего счета к среднему
    df['bill_amt_ratio'] = df['BILL_AMT1'] / (df['avg_bill_amt'] + 1)  # +1 для избежания деления на 0
    
    return df


def create_payment_amt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков на основе сумм платежей (PAY_AMT)
    
    Args:
        df: исходный DataFrame
        
    Returns:
        DataFrame с новыми признаками
    """
    df = df.copy()
    
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # Средняя сумма платежа
    df['avg_payment_amt'] = df[pay_amt_cols].mean(axis=1)
    
    # Максимальная сумма платежа
    df['max_payment_amt'] = df[pay_amt_cols].max(axis=1)
    
    # Количество месяцев с нулевым платежом
    df['num_zero_payments'] = (df[pay_amt_cols] == 0).sum(axis=1)
    
    # Тренд суммы платежа
    df['payment_amt_trend'] = df['PAY_AMT1'] - df['PAY_AMT6']
    
    return df


def create_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков использования кредита
    
    Args:
        df: исходный DataFrame
        
    Returns:
        DataFrame с новыми признаками
    """
    df = df.copy()
    
    # Коэффициент использования кредита (текущий баланс / лимит)
    df['credit_utilization'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
    
    # Средний коэффициент использования за все месяцы
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    df['avg_credit_utilization'] = df[bill_cols].mean(axis=1) / (df['LIMIT_BAL'] + 1)
    
    # Отношение суммы платежа к сумме счета
    df['payment_to_bill_ratio'] = df['PAY_AMT1'] / (df['BILL_AMT1'] + 1)
    
    # Средний payment-to-bill ratio
    payment_ratios = []
    for i in range(1, 7):
        ratio = df[f'PAY_AMT{i}'] / (df[f'BILL_AMT{i}'] + 1)
        payment_ratios.append(ratio)
    df['avg_payment_to_bill_ratio'] = pd.concat(payment_ratios, axis=1).mean(axis=1)
    
    return df


def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков на основе демографических данных
    
    Args:
        df: исходный DataFrame
        
    Returns:
        DataFrame с новыми признаками
    """
    df = df.copy()
    
    # Биннинг возраста
    df['age_group'] = pd.cut(
        df['AGE'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['young', 'adult', 'middle', 'senior', 'elderly']
    )
    
    # Отношение лимита к возрасту (условная "кредитная мощность")
    df['limit_per_age'] = df['LIMIT_BAL'] / df['AGE']
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Применение всех feature engineering трансформаций
    
    Args:
        df: исходный DataFrame
        
    Returns:
        DataFrame со всеми созданными признаками
    """
    print("🔧 Начало Feature Engineering...")
    
    # Применяем все трансформации
    df = create_payment_features(df)
    print("  ✓ Созданы признаки истории платежей")
    
    df = create_bill_features(df)
    print("  ✓ Созданы признаки счетов")
    
    df = create_payment_amt_features(df)
    print("  ✓ Созданы признаки сумм платежей")
    
    df = create_utilization_features(df)
    print("  ✓ Созданы признаки использования кредита")
    
    df = create_demographic_features(df)
    print("  ✓ Созданы демографические признаки")
    
    print(f"✅ Feature Engineering завершен. Создано {len(df.columns)} признаков")
    
    return df


def main():
    """
    Основная функция для выполнения feature engineering
    """
    # Определение путей
    project_dir = Path(__file__).resolve().parents[2]
    processed_data_dir = project_dir / 'data' / 'processed'
    
    # Обработка train данных
    print("\n📊 Обработка train данных...")
    train_path = processed_data_dir / 'train.csv'
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        train_df = engineer_features(train_df)
        train_df.to_csv(train_path, index=False)
        print(f"💾 Сохранено в {train_path}")
    else:
        print(f"❌ Файл {train_path} не найден!")
    
    # Обработка test данных
    print("\n📊 Обработка test данных...")
    test_path = processed_data_dir / 'test.csv'
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_df = engineer_features(test_df)
        test_df.to_csv(test_path, index=False)
        print(f"💾 Сохранено в {test_path}")
    else:
        print(f"❌ Файл {test_path} не найден!")
    
    print("\n✅ Feature Engineering завершен для всех данных!")


if __name__ == "__main__":
    main()
