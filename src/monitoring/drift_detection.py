"""
Модуль для мониторинга дрифта данных
Рассчитывает Population Stability Index (PSI) для обнаружения изменений в распределении данных
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import requests
import time


def calculate_psi(expected: np.array, actual: np.array, bins: int = 10) -> float:
    """
    Рассчитывает Population Stability Index (PSI)
    
    PSI - метрика для измерения изменения в распределении переменной
    PSI < 0.1: нет значительных изменений
    0.1 <= PSI < 0.2: умеренные изменения
    PSI >= 0.2: значительные изменения
    
    Args:
        expected: ожидаемое распределение (baseline, например train data)
        actual: текущее распределение (например production data)
        bins: количество бинов для гистограммы
        
    Returns:
        Значение PSI
    """
    # Определяем границы бинов на основе expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # удаляем дубликаты
    
    # Подсчет частот в бинах
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    # Нормализация к процентам
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    # Избегаем деления на ноль и логарифма от нуля
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Расчет PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi


def calculate_feature_psi(train_df: pd.DataFrame, test_df: pd.DataFrame, feature: str) -> float:
    """
    Рассчитывает PSI для конкретного признака
    
    Args:
        train_df: обучающий датасет (baseline)
        test_df: тестовый датасет (production)
        feature: имя признака
        
    Returns:
        Значение PSI для признака
    """
    # Пропускаем категориальные признаки для простоты
    if train_df[feature].dtype == 'object':
        return 0.0
    
    # Убираем NaN значения
    train_values = train_df[feature].dropna().values
    test_values = test_df[feature].dropna().values
    
    if len(train_values) == 0 or len(test_values) == 0:
        return 0.0
    
    psi = calculate_psi(train_values, test_values)
    return psi


def monitor_data_drift(
    train_path: str,
    test_path: str,
    threshold: float = 0.2
) -> dict:
    """
    Мониторинг дрифта данных путем сравнения распределений
    
    Args:
        train_path: путь к обучающим данным
        test_path: путь к тестовым/production данным
        threshold: порог для определения значительного дрифта
        
    Returns:
        Словарь с результатами мониторинга
    """
    print("🔍 Начало мониторинга дрифта данных...")
    
    # Загрузка данных
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"   Train data: {len(train_df)} записей")
    print(f"   Test data: {len(test_df)} записей")
    
    # Исключаем ID и целевую переменную
    features_to_check = [col for col in train_df.columns 
                         if col not in ['ID', 'default'] and train_df[col].dtype in ['int64', 'float64']]
    
    # Расчет PSI для каждого признака
    psi_results = {}
    drifted_features = []
    
    print(f"\n📊 Расчет PSI для {len(features_to_check)} признаков...")
    
    for feature in features_to_check:
        psi = calculate_feature_psi(train_df, test_df, feature)
        psi_results[feature] = psi
        
        if psi >= threshold:
            drifted_features.append(feature)
            print(f"   ⚠️  {feature}: PSI = {psi:.4f} (DRIFT DETECTED)")
        elif psi >= 0.1:
            print(f"   ⚡ {feature}: PSI = {psi:.4f} (moderate change)")
    
    # Общая статистика
    avg_psi = np.mean(list(psi_results.values()))
    max_psi = np.max(list(psi_results.values()))
    
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'features_checked': len(features_to_check),
        'average_psi': float(avg_psi),
        'max_psi': float(max_psi),
        'drifted_features_count': len(drifted_features),
        'drifted_features': drifted_features,
        'psi_by_feature': {k: float(v) for k, v in psi_results.items()},
        'drift_detected': len(drifted_features) > 0
    }
    
    # Вывод итогов
    print("\n" + "="*60)
    print("📈 РЕЗУЛЬТАТЫ МОНИТОРИНГА ДРИФТА")
    print("="*60)
    print(f"   Средний PSI: {avg_psi:.4f}")
    print(f"   Максимальный PSI: {max_psi:.4f}")
    print(f"   Признаков с дрифтом: {len(drifted_features)} / {len(features_to_check)}")
    
    if results['drift_detected']:
        print(f"\n   ⚠️  ОБНАРУЖЕН ДРИФТ в признаках: {', '.join(drifted_features)}")
        print("   Рекомендуется переобучить модель!")
    else:
        print("\n   ✅ Значительного дрифта не обнаружено")
    
    print("="*60 + "\n")
    
    return results


def simulate_api_monitoring(
    api_url: str,
    test_data_path: str,
    n_samples: int = 100
):
    """
    Имитация мониторинга: отправка данных на API и сбор предсказаний
    
    Args:
        api_url: URL API endpoint
        test_data_path: путь к тестовым данным
        n_samples: количество записей для отправки
    """
    print(f"🚀 Симуляция отправки данных на API: {api_url}")
    
    # Загрузка тестовых данных
    test_df = pd.read_csv(test_data_path)
    
    # Выбираем случайные записи
    sample_df = test_df.sample(n=min(n_samples, len(test_df)))
    
    # Необходимые поля для API
    required_fields = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_0', 'BILL_AMT1', 'PAY_AMT1'
    ]
    
    predictions = []
    probabilities = []
    
    print(f"📤 Отправка {len(sample_df)} запросов...")
    
    for idx, row in sample_df.iterrows():
        # Формирование payload
        payload = {field: float(row[field]) if field in row else 0.0 
                  for field in required_fields}
        
        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                predictions.append(result['default_prediction'])
                probabilities.append(result['default_probability'])
            else:
                print(f"   ⚠️  Ошибка для записи {idx}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Исключение для записи {idx}: {e}")
            continue
        
        # Небольшая задержка между запросами
        time.sleep(0.01)
    
    # Статистика предсказаний
    if predictions:
        print(f"\n✅ Получено {len(predictions)} предсказаний")
        print(f"   Средняя вероятность дефолта: {np.mean(probabilities):.4f}")
        print(f"   Предсказано дефолтов: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
    else:
        print("\n❌ Не удалось получить предсказания")


def main():
    """
    Основная функция для запуска мониторинга
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Drift Monitoring')
    parser.add_argument(
        '--mode',
        type=str,
        default='drift',
        choices=['drift', 'api'],
        help='Режим работы: drift (PSI) или api (отправка на API)'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='URL API для режима api'
    )
    
    args = parser.parse_args()
    
    # Определение путей
    project_dir = Path(__file__).resolve().parents[2]
    train_path = project_dir / 'data' / 'processed' / 'train.csv'
    test_path = project_dir / 'data' / 'processed' / 'test.csv'
    
    if args.mode == 'drift':
        # Мониторинг дрифта
        results = monitor_data_drift(str(train_path), str(test_path))
        
        # Сохранение результатов
        output_path = project_dir / 'monitoring_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Результаты сохранены в {output_path}")
        
    elif args.mode == 'api':
        # Проверка здоровья API
        try:
            response = requests.get(f"{args.api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API работает: {args.api_url}")
                simulate_api_monitoring(args.api_url, str(test_path))
            else:
                print(f"❌ API не отвечает: {response.status_code}")
        except Exception as e:
            print(f"❌ Не удалось подключиться к API: {e}")
            print(f"   Убедитесь, что API запущен на {args.api_url}")


if __name__ == "__main__":
    main()
