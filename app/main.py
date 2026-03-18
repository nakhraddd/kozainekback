from fastapi import FastAPI, WebSocket
from app.api.routes import router, manager # Import manager from routes
from app.services.camera_service import CameraService
import logging
import sys
import os
import torch
from pyngrok import ngrok
from starlette.concurrency import run_in_threadpool
import asyncio
from contextlib import asynccontextmanager
import uvicorn # Import uvicorn

# --- Setup Logging ---
# Determine the base path, which works for both normal execution and PyInstaller
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

log_file_path = os.path.join(base_dir, 'app_log.log')

# Configure logging to write to a file and the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout) # Ensure logs go to console
    ]
)
logger = logging.getLogger(__name__)

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting up application...")
    
    # Log CUDA availability
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        logger.info(f"CUDA is available! Found {count} device(s).")
        for i in range(count):
            logger.info(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA is NOT available. Application will run on CPU.")

    # Start Ngrok Tunnel
    try:
        tunnel = await run_in_threadpool(ngrok.connect, 8000, domain="anemone-intimate-utterly.ngrok-free.app")
        logger.info(f"Ngrok tunnel started at: {tunnel.public_url}")
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}", exc_info=True)

    # Start Camera Service
    logger.info("Starting camera service...")
    camera_service = CameraService()
    asyncio.create_task(camera_service.run_gui_and_stream_manager())
    logger.info("Camera service started in background.")

    yield

    # Shutdown logic
    logger.info("Shutting down application...")
    await manager.voice_assistant.shutdown()
    ngrok.kill()
    logger.info("Ngrok tunnel shut down.")

# --- FastAPI App ---
app = FastAPI(title="KOZAINEK API", lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.handle_ws(websocket)

app.include_router(router)

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        # Keep the console open for a moment to see the error if running directly
        if not getattr(sys, 'frozen', False):
            input("Press Enter to exit...")
