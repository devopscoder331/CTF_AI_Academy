FROM ubuntu:jammy-20230624

# Установка системных зависимостей точно как в рабочем варианте
RUN apt-get update -y && apt-get install gcc-11 g++-11 python3-pip curl -y

# Создание пользователя для безопасности
RUN useradd -m -u 1000 appuser

# Установка рабочей директории
WORKDIR /app

# Копирование requirements
COPY src/requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Установка llama-cpp-python точно как в рабочем варианте
RUN CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.48 --no-cache-dir

# Копирование файлов приложения
COPY src/app.py .

# Создание директории для модели
RUN mkdir -p /app/models && chown -R appuser:appuser /app

# Переключение на пользователя appuser
USER appuser

# Открытие порта
EXPOSE 8000

# Переменные окружения по умолчанию
ENV MODEL_PATH=/app/models/ggml-old-vic13b-q4_0.bin
ENV BACKEND_HOST=0.0.0.0
ENV BACKEND_PORT=8000
ENV REQUEST_TIMEOUT=30
ENV SESSION_TIMEOUT=3600
ENV CLEANUP_INTERVAL=300

# Команда запуска
CMD ["python3", "app.py"]
