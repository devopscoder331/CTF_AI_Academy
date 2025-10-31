import os
import time
import logging
from aiohttp import web, ClientSession, ClientTimeout
from aiohttp.web import Request, Response
from aiohttp.web_middlewares import middleware

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Конфигурация
FRONTEND_HOST = os.getenv('FRONTEND_HOST', '0.0.0.0')
FRONTEND_PORT = int(os.getenv('FRONTEND_PORT', '5000'))
BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:8000')
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '300'))

# Глобальная HTTP сессия
http_session: ClientSession = None


@middleware
async def error_handler(request: Request, handler):
    """Middleware для обработки ошибок"""
    try:
        return await handler(request)
    except web.HTTPException as e:
        # HTTP исключения (404, 405 и т.д.) пробрасываем как есть
        raise
    except Exception as e:
        logger.error(f"Unhandled error in {request.path}: {str(e)}", exc_info=True)
        return web.json_response(
            {'error': 'Internal server error'}, 
            status=500
        )


@middleware
async def logging_middleware(request: Request, handler):
    """Middleware для логирования запросов"""
    start_time = time.time()
    response = await handler(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.path} - "
        f"Status: {response.status} - "
        f"Time: {process_time:.3f}s"
    )
    return response


async def index(request: Request) -> Response:
    """Главная страница"""
    try:
        async with aiofiles.open('templates/index.html', 'r', encoding='utf-8') as f:
            content = await f.read()
        return web.Response(text=content, content_type='text/html')
    except Exception as e:
        logger.error(f"Error loading index.html: {str(e)}")
        return web.Response(
            text="<h1>Error loading page</h1>", 
            content_type='text/html',
            status=500
        )


async def favicon(request: Request) -> Response:
    """Обработка favicon.ico"""
    return web.Response(status=404)


async def send_message(request: Request) -> Response:
    """Прокси для отправки сообщений в backend"""
    try:
        logger.info("Received send_message request in frontend")
        data = await request.json()
        logger.debug(f"Request data: {data}")
        
        # Проксируем запрос в backend
        logger.info(f"Proxying request to {BACKEND_URL}/send_message")
        async with http_session.post(
            f"{BACKEND_URL}/send_message",
            json=data,
            timeout=ClientTimeout(total=REQUEST_TIMEOUT)
        ) as response:
            result = await response.json()
            logger.info(f"Backend response status: {response.status}")
            logger.debug(f"Backend response: {result}")
            
            if response.status >= 400:
                logger.error(f"Backend returned error: {result}")
            
            return web.json_response(result, status=response.status)
            
    except Exception as e:
        logger.error(f"Error proxying message to backend: {str(e)}", exc_info=True)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return web.json_response(
            {'error': f'Failed to send message to backend: {str(e)}'}, 
            status=500
        )


async def health_check(request: Request) -> Response:
    """Health check endpoint"""
    try:
        # Проверяем доступность backend
        async with http_session.get(
            f"{BACKEND_URL}/health",
            timeout=ClientTimeout(total=5)
        ) as response:
            backend_status = await response.json()
            
        return web.json_response({
            'status': 'healthy',
            'frontend': 'running',
            'backend': backend_status,
            'backend_url': BACKEND_URL
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return web.json_response(
            {'status': 'unhealthy', 'error': str(e)}, 
            status=500
        )


async def init_app(app):
    """Инициализация приложения"""
    global http_session
    
    # Создаем HTTP сессию для связи с backend
    timeout = ClientTimeout(total=REQUEST_TIMEOUT)
    http_session = ClientSession(timeout=timeout)
    
    logger.info("Frontend application initialized")
    logger.info(f"Backend URL: {BACKEND_URL}")


async def cleanup_app(app):
    """Очистка ресурсов приложения"""
    global http_session
    
    logger.info("Starting frontend cleanup...")
    
    if http_session:
        await http_session.close()
    
    logger.info("Frontend cleanup completed")


async def create_app():
    """Создание и настройка приложения"""
    app = web.Application(
        middlewares=[logging_middleware, error_handler]
    )
    
    # Регистрируем маршруты
    app.router.add_get('/', index)
    app.router.add_post('/send_message', send_message)
    app.router.add_get('/health', health_check)
    app.router.add_get('/favicon.ico', favicon)
    
    # Статические файлы (если понадобятся)
    app.router.add_static('/static', 'static', name='static')
    
    # Инициализация и очистка
    app.on_startup.append(init_app)
    app.on_cleanup.append(cleanup_app)
    
    return app


async def main():
    """Главная функция"""
    app = await create_app()
    
    try:
        # Запуск сервера
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, FRONTEND_HOST, FRONTEND_PORT)
        await site.start()
        
        logger.info(f"Frontend server started on {FRONTEND_HOST}:{FRONTEND_PORT}")
        logger.info(f"Backend URL: {BACKEND_URL}")
        
        # Ждем бесконечно
        await asyncio.Future()
        
    except Exception as e:
        logger.error(f"Error starting frontend server: {str(e)}")
        raise
    finally:
        logger.info("Shutting down frontend server...")
        await runner.cleanup()


if __name__ == '__main__':
    import asyncio
    import time
    import aiofiles
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Frontend server stopped by user")
    except Exception as e:
        logger.error(f"Frontend server error: {str(e)}")
        raise
