# Итоговый проект: Credit Scoring Model Pipeline
## Автоматизация процессов разработки и тестирования моделей машинного обучения

---

## 📊 Краткое описание

Полностью автоматизированный end-to-end ML пайплайн для предсказания дефолта клиентов по кредитным картам.

**Датасет:** UCI Credit Card Default Dataset (30,000 записей, 24 признака)

**Задача:** Бинарная классификация (предсказание дефолта)

---

## ✅ Реализованные требования 

### 1. Организация кода и Git 
- ✅ Структурированный репозиторий с папками src/, tests/, notebooks/, data/, models/
- ✅ Чистая структура с использованием best practices
- ✅ Подробный README.md с инструкциями
- ✅ .gitignore для правильного игнорирования файлов

**Файлы:**
- `README.md` - основная документация
- `GETTING_STARTED.md` - пошаговое руководство
- `.gitignore` - правила игнорирования
- Вся структура папок организована по стандарту

### 2. Подготовка и валидация данных 
- ✅ EDA в Jupyter notebook
- ✅ Скрипт загрузки и очистки данных
- ✅ Feature Engineering с созданием 15+ новых признаков
- ✅ Great Expectations для валидации данных (10 правил валидации)
- ✅ Тесты валидации интегрированы в пайплайн

**Файлы:**
- `src/data/make_dataset.py` - загрузка и очистка данных
- `src/data/validation.py` - валидация с Great Expectations
- `src/features/build_features.py` - feature engineering
- `notebooks/eda.ipynb` - исследовательский анализ
- `tests/test_data.py` - unit-тесты для данных

**Созданные признаки:**
- Признаки истории платежей (avg_payment_delay, max_payment_delay, num_months_delayed, has_payment_delay, payment_trend)
- Признаки счетов (avg_bill_amt, max_bill_amt, std_bill_amt, bill_trend, bill_amt_ratio)
- Признаки платежей (avg_payment_amt, max_payment_amt, num_zero_payments, payment_amt_trend)
- Признаки использования кредита (credit_utilization, avg_credit_utilization, payment_to_bill_ratio, avg_payment_to_bill_ratio)
- Демографические признаки (age_group, limit_per_age)

### 3. Построение и настройка модели 
- ✅ Sklearn Pipeline с предобработкой (Imputer, Scaler, OneHotEncoder)
- ✅ Поддержка GradientBoosting и LogisticRegression
- ✅ GridSearchCV для подбора гиперпараметров
- ✅ Все ключевые метрики: ROC-AUC, Precision, Recall, F1-Score
- ✅ ROC-кривая и Confusion Matrix
- ✅ Стратифицированное разделение данных (80/20)

**Файлы:**
- `src/models/pipeline.py` - создание sklearn Pipeline
- `src/models/train.py` - обучение с метриками
- `src/models/predict.py` - inference
- `tests/test_models.py` - unit-тесты для моделей
- `tests/test_features.py` - unit-тесты для признаков

### 4. MLflow Tracking 
- ✅ Полная интеграция MLflow в код обучения
- ✅ Логирование всех параметров модели
- ✅ Логирование всех метрик (ROC-AUC, Precision, Recall, F1)
- ✅ Логирование артефактов (ROC-кривая, Confusion Matrix, модель)
- ✅ Поддержка множественных экспериментов (GradientBoosting, LogisticRegression, с/без GridSearch)
- ✅ Автоматическое сохранение модели через mlflow.sklearn.log_model

**Файлы:**
- `src/models/train.py` - интеграция MLflow (строки с mlflow.*)
- MLflow UI доступен через: `mlflow ui`

**Примеры экспериментов для проведения:**
1. GradientBoosting без GridSearch
2. GradientBoosting с GridSearch
3. LogisticRegression без GridSearch
4. LogisticRegression с GridSearch
5. GradientBoosting с другими параметрами

### 5. DVC 
- ✅ DVC инициализирован в репозитории
- ✅ Версионирование данных (data/raw/, data/processed/)
- ✅ Версионирование моделей (models/)
- ✅ DVC pipeline в dvc.yaml с 3 стадиями (prepare, feature_engineering, train)
- ✅ Возможность воспроизведения через `dvc repro`

**Файлы:**
- `dvc.yaml` - конфигурация DVC pipeline
- `.dvc/.gitignore` - создается при инициализации DVC

**Команды:**
```bash
dvc init
dvc add data/raw/UCI_Credit_Card.csv
dvc add models/
dvc repro  # Воспроизведение пайплайна
```

### 6. Тестирование и CI 
- ✅ Unit-тесты с pytest для всех модулей
- ✅ Тесты для data processing (5 тестов)
- ✅ Тесты для feature engineering (6 тестов)
- ✅ Тесты для моделей (5 тестов)
- ✅ GitHub Actions CI/CD workflow
- ✅ Линтинг с flake8
- ✅ Форматирование с black
- ✅ Валидация данных интегрирована в CI
- ✅ Coverage report

**Файлы:**
- `tests/test_data.py` - 5 unit-тестов
- `tests/test_features.py` - 6 unit-тестов
- `tests/test_models.py` - 5 unit-тестов
- `.github/workflows/ci-cd.yml` - CI/CD конфигурация
- `pyproject.toml` - конфигурация инструментов

**Запуск:**
```bash
pytest tests/ -v --cov=src
black src/ tests/
flake8 src/ tests/
```

### 7. Docker и FastAPI 
- ✅ Корректный Dockerfile
- ✅ FastAPI приложение с endpoint /predict
- ✅ Pydantic модели для валидации входных данных
- ✅ Endpoint /health для health check
- ✅ Автоматическая документация Swagger UI
- ✅ Возвращает класс и вероятность дефолта
- ✅ Обработка ошибок

**Файлы:**
- `Dockerfile` - конфигурация Docker
- `src/api/app.py` - FastAPI приложение

**Endpoints:**
- `GET /` - корневой endpoint
- `GET /health` - проверка здоровья
- `POST /predict` - предсказание дефолта
- `GET /docs` - Swagger UI документация

**Запуск:**
```bash
# Локально
uvicorn src.api.app:app --reload

# Docker
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api
```

### 8. Мониторинг дрифта 
- ✅ Скрипт для расчета Population Stability Index (PSI)
- ✅ Сравнение распределений train vs test данных
- ✅ Расчет PSI для всех числовых признаков
- ✅ Определение дрифта (PSI > 0.2)
- ✅ Имитация отправки данных на API
- ✅ Сохранение результатов в JSON

**Файлы:**
- `src/monitoring/drift_detection.py` - детекция дрифта

**Запуск:**
```bash
# PSI мониторинг
python src/monitoring/drift_detection.py --mode drift

# API мониторинг
python src/monitoring/drift_detection.py --mode api --api-url http://localhost:8000
```

### 9. Демонстрация 
- ✅ Готов к презентации
- ✅ Все компоненты протестированы
- ✅ Документация подробная
- ✅ Пошаговое руководство GETTING_STARTED.md

---

## 📁 Полная структура проекта

```
credit-scoring-model/
│
├── data/                               # Данные (версионируется DVC)
│   ├── raw/                           # Исходные данные
│   │   └── UCI_Credit_Card.csv       # 2.8MB датасет
│   ├── processed/                     # Обработанные данные
│   │   ├── train.csv                 # После make_dataset.py и build_features.py
│   │   └── test.csv                  # После make_dataset.py и build_features.py
│   └── expectations/                  # Great Expectations
│
├── models/                            # Обученные модели (версионируется DVC)
│   ├── credit_default_model_gradient_boosting.pkl
│   ├── credit_default_model_logistic_regression.pkl
│   └── plots/                        # Графики ROC, Confusion Matrix
│
├── notebooks/                         # Jupyter notebooks
│   └── eda.ipynb                     # Exploratory Data Analysis
│
├── src/                               # Исходный код
│   ├── __init__.py
│   │
│   ├── data/                         # Обработка данных
│   │   ├── __init__.py
│   │   ├── make_dataset.py          # Загрузка и очистка (4KB)
│   │   └── validation.py            # Great Expectations (9KB)
│   │
│   ├── features/                     # Feature Engineering
│   │   ├── __init__.py
│   │   └── build_features.py        # Создание признаков (7KB)
│   │
│   ├── models/                       # ML модели
│   │   ├── __init__.py
│   │   ├── pipeline.py              # Sklearn Pipeline (3KB)
│   │   ├── train.py                 # Обучение с MLflow (11KB)
│   │   └── predict.py               # Inference (3KB)
│   │
│   ├── api/                          # FastAPI
│   │   ├── __init__.py
│   │   └── app.py                   # REST API (6KB)
│   │
│   └── monitoring/                   # Мониторинг
│       ├── __init__.py
│       └── drift_detection.py       # PSI расчет (9KB)
│
├── tests/                             # Unit-тесты
│   ├── __init__.py
│   ├── test_data.py                  # Тесты данных (3.5KB)
│   ├── test_features.py              # Тесты признаков (5.5KB)
│   └── test_models.py                # Тесты моделей (7KB)
│
├── .github/                          # GitHub Actions
│   └── workflows/
│       └── ci-cd.yml                 # CI/CD pipeline (3KB)
│
├── .gitignore                        # Git ignore rules (0.5KB)
├── Dockerfile                        # Docker конфигурация (1KB)
├── README.md                         # Основная документация (8.5KB)
├── GETTING_STARTED.md                # Пошаговое руководство (9.5KB)
├── PROJECT_SUMMARY.md                # Этот файл (резюме проекта)
├── requirements.txt                  # Python зависимости (0.5KB)
├── pyproject.toml                    # Конфигурация инструментов (1KB)
├── dvc.yaml                          # DVC pipeline (1.5KB)
└── run_pipeline.sh                   # Скрипт запуска (3KB)
```

---

## 🚀 Быстрый запуск (3 команды)

```bash
# 1. Установка зависимостей
pip install -r requirements.txt

# 2. Запуск пайплайна
./run_pipeline.sh  # Linux/Mac
# или вручную для Windows: python src/data/make_dataset.py && ...

# 3. Просмотр результатов
mlflow ui
```

---

## 📊 Метрики качества

После обучения модели вы получите:
- **ROC-AUC:** ~0.75-0.78
- **Precision:** ~0.65-0.70
- **Recall:** ~0.40-0.50
- **F1-Score:** ~0.50-0.55

*(Точные значения зависят от параметров модели)*

---

## 🎯 Ключевые технологии

- **ML:** scikit-learn, pandas, numpy
- **Experiment Tracking:** MLflow
- **Data Versioning:** DVC
- **Data Validation:** Great Expectations
- **API:** FastAPI, Pydantic, Uvicorn
- **Testing:** pytest, pytest-cov
- **CI/CD:** GitHub Actions
- **Code Quality:** black, flake8
- **Containerization:** Docker
- **Visualization:** matplotlib, seaborn

---

## 📈 Проведение экспериментов 

Рекомендуется провести минимум 5 экспериментов:

```bash
# Эксперимент 1: GradientBoosting базовый
python src/models/train.py --model gradient_boosting

# Эксперимент 2: GradientBoosting с GridSearch
python src/models/train.py --model gradient_boosting --grid-search

# Эксперимент 3: LogisticRegression базовый
python src/models/train.py --model logistic_regression

# Эксперимент 4: LogisticRegression с GridSearch
python src/models/train.py --model logistic_regression --grid-search

# Эксперимент 5: Измените параметры в train.py и повторите
# Например, измените n_estimators, learning_rate, max_depth
```

Все эксперименты будут видны в MLflow UI для сравнения.

---

## 📞 Поддержка

Если возникнут вопросы:
1. Смотрите `GETTING_STARTED.md` - пошаговое руководство
2. Смотрите `README.md` - техническая документация
3. Смотрите комментарии в коде - каждая функция задокументирована

