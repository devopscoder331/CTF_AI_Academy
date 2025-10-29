FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя для безопасности
RUN useradd -m -u 1000 appuser

# Установка рабочей директории
WORKDIR /app

# Копирование requirements
COPY src/requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов приложения
COPY src/app.py .
COPY src/templates/ ./templates/

# Создание директории для статических файлов
RUN mkdir -p /app/static && chown -R appuser:appuser /app

# Переключение на пользователя appuser
USER appuser

# Открытие порта
EXPOSE 5000

# Переменные окружения по умолчанию
ENV FRONTEND_HOST=0.0.0.0
ENV FRONTEND_PORT=5000
ENV BACKEND_URL=http://backend:8000
ENV REQUEST_TIMEOUT=30

# Команда запуска
CMD ["python", "app.py"]