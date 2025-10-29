# AI Chat Bot - Микросервисное приложение

Это микросервисное приложение для AI чат-бота с асинхронной архитектурой.

## Архитектура

- **Frontend** (порт 5000) - веб-интерфейс для пользователей
- **Backend** (порт 8000) - API сервис с асинхронной обработкой

## Особенности Backend

- ✅ Асинхронная архитектура с aiohttp
- ✅ Connection pooling и управление сессиями
- ✅ Graceful shutdown с обработкой сигналов
- ✅ Timeout handling и error handling
- ✅ Session management с автоматической очисткой
- ✅ Health check endpoints
- ✅ Middleware для логирования и обработки ошибок
- ✅ Автоматическая очистка истекших сессий

## Переменные окружения

### Backend
- `MODEL_PATH` - путь к модели внутри контейнера (по умолчанию: `/app/models/ggml-old-vic13b-q4_0.bin`)
- `BACKEND_HOST` - хост для backend (по умолчанию: `0.0.0.0`)
- `BACKEND_PORT` - порт для backend (по умолчанию: `8000`)
- `REQUEST_TIMEOUT` - таймаут запросов в секундах (по умолчанию: `30`)
- `SESSION_TIMEOUT` - таймаут сессии в секундах (по умолчанию: `3600`)
- `CLEANUP_INTERVAL` - интервал очистки сессий в секундах (по умолчанию: `300`)

### Frontend
- `FRONTEND_HOST` - хост для frontend (по умолчанию: `0.0.0.0`)
- `FRONTEND_PORT` - порт для frontend (по умолчанию: `5000`)
- `BACKEND_URL` - URL backend сервиса (по умолчанию: `http://backend:8000`)
- `REQUEST_TIMEOUT` - таймаут запросов в секундах (по умолчанию: `30`)

## Запуск

### 1. Создайте .env файл (опционально)

```bash
# В директории compose/
cat > .env << EOF
MODEL_PATH=/app/models/ggml-old-vic13b-q4_0.bin
MODEL_HOST_PATH=/home/luksa/git/AI_Workshop/ai-goat/tasks/models
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
REQUEST_TIMEOUT=30
SESSION_TIMEOUT=3600
CLEANUP_INTERVAL=300
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=5000
BACKEND_URL=http://backend:8000
EOF
```

### 2. Запустите приложение

```bash
# Перейдите в директорию compose
cd /home/luksa/git/AI_Workshop/ai-goat/tasks/task_1/compose/

# Запустите все сервисы
docker-compose up --build

# Или в фоновом режиме
docker-compose up -d --build
```

### 3. Проверьте работу

- Frontend: http://localhost:5000
- Backend Health: http://localhost:8000/health
- Frontend Health: http://localhost:5000/health

## Смена модели

Чтобы использовать другую модель, измените переменную окружения `MODEL_HOST_PATH` в docker-compose.yml или .env файле:

```yaml
environment:
  - MODEL_PATH=/app/models/your-new-model.bin
volumes:
  - /path/to/your/models:/app/models:ro
```

## API Endpoints

### Backend

- `GET /health` - проверка состояния сервиса
- `POST /send_message` - отправка сообщения
  ```json
  {
    "message": "Привет!",
    "session_id": "optional_session_id"
  }
  ```
- `GET /session/{session_id}` - информация о сессии

### Frontend

- `GET /` - главная страница
- `POST /send_message` - прокси для отправки сообщений в backend
- `GET /health` - проверка состояния frontend и backend

## Мониторинг

Приложение включает:
- Health checks для всех сервисов
- Логирование всех запросов
- Автоматическая очистка истекших сессий
- Graceful shutdown при получении сигналов SIGTERM/SIGINT

## Остановка

```bash
# Остановка сервисов
docker-compose down

# Остановка с удалением volumes
docker-compose down -v
```
