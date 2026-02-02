FROM python:3.9-slim

# Установка рабочей директории
WORKDIR /app

# Копирование файла с зависимостями
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ ./src/
COPY models/ ./models/

# Открытие порта
EXPOSE 8000

# Команда запуска приложения
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
