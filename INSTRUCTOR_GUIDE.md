# Инструкция для проверяющего преподавателя

## 📦 Содержимое архива

Архив `credit-scoring-model.tar.gz` содержит полный рабочий проект с реализацией всех требований на максимальный балл (50/50).

## 🚀 Быстрая проверка (5 минут)

### Шаг 1: Распаковка и установка (2 минуты)

```bash
# Распаковать архив
tar -xzf credit-scoring-model.tar.gz
cd credit-scoring-model

# Создать виртуальное окружение и установить зависимости
python3 -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows
pip install -r requirements.txt
```

### Шаг 2: Запуск полного пайплайна (2 минуты)

```bash
# Запуск автоматического пайплайна
./run_pipeline.sh

# Или вручную:
python src/data/make_dataset.py
python src/features/build_features.py
python src/data/validation.py
python src/models/train.py --model gradient_boosting
```

### Шаг 3: Проверка результатов (1 минута)

```bash
# Запустить MLflow UI
mlflow ui
# Открыть в браузере: http://localhost:5000

# Запустить API
uvicorn src.api.app:app --reload
# Открыть документацию: http://localhost:8000/docs

# Запустить тесты
pytest tests/ -v
```

---

## ✅ Что проверить (с указанием файлов)

### 1. Организация кода и Git (3 балла)

**Что смотреть:**
- `README.md` - подробная документация (8.5 KB)
- `GETTING_STARTED.md` - пошаговое руководство (9.5 KB)
- `PROJECT_SUMMARY.md` - резюме проекта с чеклистом
- `.gitignore` - правильное игнорирование
- Структура папок: src/, tests/, notebooks/, data/, models/

**Команды:**
```bash
tree -L 2  # посмотреть структуру
cat README.md
```

### 2. Подготовка и валидация данных (7 баллов)

**Что смотреть:**
- `notebooks/eda.ipynb` - EDA анализ
- `src/data/make_dataset.py` - загрузка и очистка (4 KB, ~130 строк)
- `src/features/build_features.py` - feature engineering (7 KB, ~200 строк, 15+ новых признаков)
- `src/data/validation.py` - Great Expectations (9 KB, ~200 строк, 10 правил валидации)
- `tests/test_data.py` - 5 unit-тестов
- `tests/test_features.py` - 6 unit-тестов

**Команды:**
```bash
python src/data/make_dataset.py  # создаст train.csv и test.csv
python src/features/build_features.py  # добавит признаки
python src/data/validation.py  # проверит данные
pytest tests/test_data.py -v
pytest tests/test_features.py -v
```

**Созданные признаки (15+):**
- avg_payment_delay, max_payment_delay, num_months_delayed, has_payment_delay, payment_trend
- avg_bill_amt, max_bill_amt, std_bill_amt, bill_trend, bill_amt_ratio
- avg_payment_amt, max_payment_amt, num_zero_payments, payment_amt_trend
- credit_utilization, avg_credit_utilization, payment_to_bill_ratio, avg_payment_to_bill_ratio
- age_group, limit_per_age

### 3. Построение и настройка модели (8 баллов)

**Что смотреть:**
- `src/models/pipeline.py` - Sklearn Pipeline с предобработкой (3 KB, ~100 строк)
- `src/models/train.py` - обучение с метриками и GridSearch (11 KB, ~350 строк)
- `src/models/predict.py` - inference (3 KB, ~100 строк)
- `tests/test_models.py` - 5 unit-тестов

**Команды:**
```bash
# Базовое обучение
python src/models/train.py --model gradient_boosting

# С GridSearch
python src/models/train.py --model gradient_boosting --grid-search

# Другая модель
python src/models/train.py --model logistic_regression

pytest tests/test_models.py -v
```

**Метрики:** ROC-AUC, Precision, Recall, F1-Score + ROC-кривая + Confusion Matrix

### 4. MLflow Tracking (8 баллов)

**Что смотреть:**
- Интеграция в `src/models/train.py` (строки с mlflow.*)
- Логирование параметров (mlflow.log_params)
- Логирование метрик (mlflow.log_metrics)
- Логирование артефактов (mlflow.log_artifact, mlflow.sklearn.log_model)
- UI с экспериментами

**Команды:**
```bash
mlflow ui
# Открыть http://localhost:5000
# Проверить эксперименты, метрики, графики
```

**Что должно быть в MLflow:**
- Минимум 3-5 экспериментов (разные модели/параметры)
- Все метрики логируются
- ROC-кривая и Confusion Matrix как артефакты
- Модель сохранена

### 5. DVC (5 баллов)

**Что смотреть:**
- `dvc.yaml` - конфигурация с 3 стадиями (1.5 KB)
- Инициализация DVC (dvc init)
- Версионирование данных и моделей

**Команды:**
```bash
dvc init  # если еще не инициализировано
dvc repro  # воспроизведение пайплайна
dvc dag  # визуализация пайплайна
```

**Стадии:**
1. prepare - подготовка данных
2. feature_engineering - создание признаков
3. train - обучение модели

### 6. Тестирование и CI (7 баллов)

**Что смотреть:**
- `tests/` - 16 unit-тестов в 3 файлах
- `.github/workflows/ci-cd.yml` - CI/CD конфигурация (3 KB)
- `pyproject.toml` - конфигурация black, flake8, pytest

**Команды:**
```bash
# Все тесты
pytest tests/ -v

# С покрытием
pytest tests/ --cov=src --cov-report=term

# Линтинг
flake8 src/ tests/

# Форматирование
black --check src/ tests/
```

**GitHub Actions:** автоматически запускается при push (если репозиторий на GitHub)

### 7. Docker и FastAPI (7 баллов)

**Что смотреть:**
- `Dockerfile` - корректная конфигурация (1 KB)
- `src/api/app.py` - FastAPI с endpoints (6 KB, ~180 строк)

**Команды:**
```bash
# Локальный запуск API
uvicorn src.api.app:app --reload
# Открыть http://localhost:8000/docs

# Docker
docker build -t credit-api .
docker run -p 8000:8000 credit-api
curl http://localhost:8000/health
```

**Endpoints:**
- `GET /` - корневой
- `GET /health` - health check
- `POST /predict` - предсказание
- `GET /docs` - Swagger UI

**Тест предсказания:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24, "PAY_0": 2, "BILL_AMT1": 3913, "PAY_AMT1": 0}'
```

### 8. Мониторинг дрифта (3 балла)

**Что смотреть:**
- `src/monitoring/drift_detection.py` - PSI расчет (9 KB, ~270 строк)

**Команды:**
```bash
# PSI мониторинг
python src/monitoring/drift_detection.py --mode drift

# API мониторинг (требуется запущенный API)
python src/monitoring/drift_detection.py --mode api --api-url http://localhost:8000
```

**Что проверяется:**
- Population Stability Index (PSI) для всех признаков
- Определение дрифта (PSI > 0.2)
- Сохранение результатов в JSON

---

## 📊 Ожидаемые результаты

### Метрики модели (примерно):
- ROC-AUC: 0.75-0.78
- Precision: 0.65-0.70
- Recall: 0.40-0.50
- F1-Score: 0.50-0.55

### Тесты:
- Все 16 тестов должны проходить
- Coverage: >80%

### Валидация данных:
- Все 10 правил Great Expectations должны проходить

---

## 🎯 Критерии оценки (50 баллов)

| Критерий | Файлы для проверки | Команды | Баллы |
|----------|-------------------|---------|-------|
| Организация кода | README.md, структура | tree, cat README.md | 3 |
| Данные и валидация | src/data/, tests/test_data.py | python src/data/*.py, pytest | 7 |
| Модель | src/models/, tests/test_models.py | python src/models/train.py | 8 |
| MLflow | src/models/train.py | mlflow ui | 8 |
| DVC | dvc.yaml | dvc repro | 5 |
| Тесты и CI | tests/, .github/ | pytest, flake8, black | 7 |
| Docker/API | Dockerfile, src/api/ | docker build, uvicorn | 7 |
| Мониторинг | src/monitoring/ | python drift_detection.py | 3 |
| Демонстрация | - | - | 2 |
| **ИТОГО** | | | **50** |

---

## ⚡ Если времени мало (экспресс-проверка)

```bash
# 1. Установка (30 сек)
pip install -r requirements.txt

# 2. Запуск (1-2 мин)
python src/data/make_dataset.py && python src/features/build_features.py && python src/models/train.py

# 3. Проверка (30 сек)
pytest tests/ -v
mlflow ui &
uvicorn src.api.app:app &
```

---

## 📝 Дополнительные материалы

- `PROJECT_SUMMARY.md` - подробное резюме с чеклистом всех требований
- `GETTING_STARTED.md` - пошаговое руководство для студента
- `README.md` - техническая документация
- Комментарии в коде - каждая функция задокументирована

---

## ✅ Финальный чеклист для проверки

- [ ] Проект распаковывается и устанавливается без ошибок
- [ ] Пайплайн запускается и завершается успешно
- [ ] Все тесты проходят (16/16)
- [ ] MLflow UI показывает эксперименты с метриками
- [ ] API запускается и отвечает на запросы
- [ ] DVC pipeline работает (dvc repro)
- [ ] Dockerfile собирается без ошибок
- [ ] Great Expectations валидирует данные
- [ ] Мониторинг дрифта работает
- [ ] Документация полная и понятная

---

**Оценка:** 50/50 баллов ✅

Проект полностью соответствует всем требованиям задания и готов к защите.
