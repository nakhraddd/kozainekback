import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import websockets
import logging
import torch
from pyngrok import ngrok
from app.api.routes import manager
from app.services.camera_service import CameraService

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

log_file_path = os.path.join(base_dir, 'app_log.log')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


async def handle_client(websocket):
    try:
        # Теперь мы передаем чистый websocket напрямую, без адаптеров
        await manager.handle_ws(websocket)
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        logger.error(f"Client connection error: {e}")


async def main():
    logger.info("Starting up application...")

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        logger.info(f"CUDA is available! Found {count} device(s).")
        for i in range(count):
            logger.info(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA is NOT available. Application will run on CPU.")

    try:
        tunnel = await asyncio.to_thread(ngrok.connect, 8000, domain="anemone-intimate-utterly.ngrok-free.app")
        logger.info(f"Ngrok tunnel started at: {tunnel.public_url}")
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}", exc_info=True)

    logger.info("Starting camera service...")
    camera_service = CameraService()
    camera_task = asyncio.create_task(camera_service.run_gui_and_stream_manager())
    logger.info("Camera service started in background.")

    server = await websockets.serve(handle_client, "0.0.0.0", 8000)

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down application...")
        server.close()
        await server.wait_closed()
        if hasattr(manager, 'voice_assistant'):
            await manager.voice_assistant.shutdown()
        ngrok.kill()
        camera_task.cancel()
        logger.info("Ngrok tunnel shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        if not getattr(sys, 'frozen', False):
            input("Press Enter to exit...")