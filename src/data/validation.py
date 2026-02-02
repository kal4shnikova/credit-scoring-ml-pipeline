"""
Модуль для валидации данных с помощью Great Expectations
"""
import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration
import pandas as pd
from pathlib import Path
import sys


def create_expectation_suite(context: gx.DataContext) -> gx.core.ExpectationSuite:
    """
    Создание набора правил валидации для датасета кредитных карт
    
    Args:
        context: Great Expectations DataContext
        
    Returns:
        ExpectationSuite с правилами валидации
    """
    suite_name = "credit_data_suite"
    
    # Создание или получение существующего suite
    try:
        suite = context.get_expectation_suite(suite_name)
        print(f"Использую существующий suite: {suite_name}")
    except:
        suite = context.add_expectation_suite(expectation_suite_name=suite_name)
        print(f"Создан новый suite: {suite_name}")
    
    # Список ожидаемых правил валидации
    expectations = [
        # 1. Проверка наличия всех необходимых колонок
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_set",
            kwargs={
                "column_set": [
                    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
                    "default"
                ],
                "exact_match": False
            }
        ),
        
        # 2. Проверка, что LIMIT_BAL не содержит null
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "LIMIT_BAL"}
        ),
        
        # 3. Проверка диапазона значений для AGE (18-100 лет)
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "AGE",
                "min_value": 18,
                "max_value": 100
            }
        ),
        
        # 4. Проверка, что SEX принимает значения 1 или 2
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "SEX",
                "value_set": [1, 2]
            }
        ),
        
        # 5. Проверка, что EDUCATION принимает значения 1, 2, 3, 4
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "EDUCATION",
                "value_set": [1, 2, 3, 4]
            }
        ),
        
        # 6. Проверка, что MARRIAGE принимает значения 1, 2, 3
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "MARRIAGE",
                "value_set": [1, 2, 3]
            }
        ),
        
        # 7. Проверка, что целевая переменная бинарная (0 или 1)
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "default",
                "value_set": [0, 1]
            }
        ),
        
        # 8. Проверка, что LIMIT_BAL положительный
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "LIMIT_BAL",
                "min_value": 0,
                "max_value": 1000000
            }
        ),
        
        # 9. Проверка типов данных для числовых колонок
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={
                "column": "AGE",
                "type_": "int64"
            }
        ),
        
        # 10. Проверка на отсутствие дубликатов по ID
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "ID"}
        ),
    ]
    
    # Добавление всех правил в suite
    for expectation in expectations:
        suite.add_expectation_configuration(expectation)
    
    # Сохранение suite
    context.add_or_update_expectation_suite(expectation_suite=suite)
    
    print(f"✅ Создано {len(expectations)} правил валидации")
    return suite


def validate_data(data_path: str, context: gx.DataContext, suite_name: str = "credit_data_suite") -> bool:
    """
    Валидация данных с использованием созданного suite
    
    Args:
        data_path: путь к CSV файлу с данными
        context: Great Expectations DataContext
        suite_name: имя expectation suite
        
    Returns:
        True если валидация прошла успешно, False иначе
    """
    print(f"\n🔍 Валидация данных из: {data_path}")
    
    # Загрузка данных
    df = pd.read_csv(data_path)
    
    # Если есть колонка 'default.payment.next.month', переименовываем её
    if 'default.payment.next.month' in df.columns:
        df = df.rename(columns={'default.payment.next.month': 'default'})
    
    # Создание Batch для валидации
    batch = context.sources.add_pandas("pandas_datasource").add_dataframe_asset(
        name="credit_data"
    ).build_batch_request(dataframe=df)
    
    # Получение suite
    suite = context.get_expectation_suite(suite_name)
    
    # Создание Validator
    validator = context.get_validator(
        batch_request=batch,
        expectation_suite=suite
    )
    
    # Выполнение валидации
    results = validator.validate()
    
    # Вывод результатов
    print(f"\n📊 Результаты валидации:")
    print(f"   Всего проверок: {results.statistics['evaluated_expectations']}")
    print(f"   Успешно: {results.statistics['successful_expectations']}")
    print(f"   Провалено: {results.statistics['unsuccessful_expectations']}")
    print(f"   Процент успеха: {results.statistics['success_percent']:.2f}%")
    
    # Вывод деталей провалившихся проверок
    if not results.success:
        print("\n❌ Провалившиеся проверки:")
        for result in results.results:
            if not result.success:
                expectation_type = result.expectation_config.expectation_type
                column = result.expectation_config.kwargs.get('column', 'N/A')
                print(f"   - {expectation_type} для колонки '{column}'")
                if 'observed_value' in result.result:
                    print(f"     Наблюдаемое значение: {result.result['observed_value']}")
    else:
        print("\n✅ Все проверки пройдены успешно!")
    
    return results.success


def main():
    """
    Основная функция для выполнения валидации данных
    """
    # Определение путей
    project_dir = Path(__file__).resolve().parents[2]
    ge_dir = project_dir / 'data' / 'expectations'
    processed_data_dir = project_dir / 'data' / 'processed'
    
    # Создание директории для Great Expectations если не существует
    ge_dir.mkdir(parents=True, exist_ok=True)
    
    # Инициализация DataContext
    try:
        context = gx.get_context(project_root_dir=str(ge_dir))
    except:
        context = gx.get_context(mode="file", project_root_dir=str(ge_dir))
    
    print("✅ Great Expectations DataContext инициализирован")
    
    # Создание expectation suite
    suite = create_expectation_suite(context)
    
    # Валидация train данных
    train_path = processed_data_dir / 'train.csv'
    if train_path.exists():
        train_valid = validate_data(str(train_path), context)
    else:
        print(f"⚠️ Файл {train_path} не найден. Запустите сначала make_dataset.py")
        train_valid = False
    
    # Валидация test данных
    test_path = processed_data_dir / 'test.csv'
    if test_path.exists():
        test_valid = validate_data(str(test_path), context)
    else:
        print(f"⚠️ Файл {test_path} не найден. Запустите сначала make_dataset.py")
        test_valid = False
    
    # Выход с кодом ошибки если валидация провалилась
    if not (train_valid and test_valid):
        print("\n❌ Валидация данных провалена!")
        sys.exit(1)
    else:
        print("\n✅ Валидация всех данных успешно завершена!")
        sys.exit(0)


if __name__ == "__main__":
    main()
