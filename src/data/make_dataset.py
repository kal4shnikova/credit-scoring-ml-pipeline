"""
Модуль для загрузки и первичной обработки датасета UCI Credit Card
"""
import pandas as pd
import os
from pathlib import Path


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Загрузка исходного датасета
    
    Args:
        filepath: путь к CSV файлу
        
    Returns:
        DataFrame с исходными данными
    """
    print(f"Загрузка данных из {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Загружено {len(df)} записей, {len(df.columns)} признаков")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Первичная очистка данных
    
    Args:
        df: исходный DataFrame
        
    Returns:
        Очищенный DataFrame
    """
    print("Выполнение первичной очистки данных...")
    
    # Создаем копию для избежания SettingWithCopyWarning
    df = df.copy()
    
    # Удаление дубликатов
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Удалено {initial_rows - len(df)} дубликатов")
    
    # Удаление строк с пропущенными значениями (если есть)
    initial_rows = len(df)
    df = df.dropna()
    print(f"Удалено {initial_rows - len(df)} строк с пропусками")
    
    # Переименование целевой переменной для удобства
    df = df.rename(columns={'default.payment.next.month': 'default'})
    
    # Корректировка аномальных значений в EDUCATION
    # 0, 5, 6 -> категория "other" (4)
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    df.loc[df['EDUCATION'] < 1, 'EDUCATION'] = 4
    
    # Корректировка аномальных значений в MARRIAGE
    # 0 -> категория "other" (3)
    df.loc[df['MARRIAGE'] == 0, 'MARRIAGE'] = 3
    
    print("Первичная очистка завершена")
    return df


def split_and_save_data(df: pd.DataFrame, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """
    Разделение данных на train/test и сохранение
    
    Args:
        df: DataFrame для разделения
        output_dir: директория для сохранения
        test_size: доля тестовой выборки
        random_state: seed для воспроизводимости
    """
    from sklearn.model_selection import train_test_split
    
    print(f"Разделение данных на train/test (test_size={test_size})...")
    
    # Разделение на train и test
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['default']  # стратификация по целевой переменной
    )
    
    # Создание директории если не существует
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Сохранение
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train set: {len(train_df)} записей -> {train_path}")
    print(f"Test set: {len(test_df)} записей -> {test_path}")
    print(f"Распределение классов в train: {train_df['default'].value_counts().to_dict()}")
    print(f"Распределение классов в test: {test_df['default'].value_counts().to_dict()}")


def main():
    """
    Основная функция для выполнения всего пайплайна подготовки данных
    """
    # Определение путей
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_path = project_dir / 'data' / 'raw' / 'UCI_Credit_Card.csv'
    processed_data_dir = project_dir / 'data' / 'processed'
    
    # Загрузка исходных данных
    df = load_raw_data(raw_data_path)
    
    # Очистка данных
    df_clean = clean_data(df)
    
    # Разделение и сохранение
    split_and_save_data(df_clean, processed_data_dir)
    
    print("\n✅ Подготовка данных завершена успешно!")


if __name__ == "__main__":
    main()
