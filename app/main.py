from fastapi import FastAPI, WebSocket
from app.api.routes import router, ConnectionManager
from app.services.detector import YoloDetector
from app.services.camera_service import CameraService
import logging
from pyngrok import ngrok
from starlette.concurrency import run_in_threadpool
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="KOZAINEK API")

detector = YoloDetector(model_path="yolov8n-seg.pt")
manager = ConnectionManager(detector=detector)

camera_service = CameraService()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.handle_ws(websocket)

app.include_router(router)

@app.on_event("startup")
async def start_ngrok_tunnel():
    try:
        tunnel = await run_in_threadpool(ngrok.connect, 8000, domain="anemone-intimate-utterly.ngrok-free.app")
        logger.info(f"Ngrok tunnel started at: {tunnel.public_url}")
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}")

@app.on_event("startup")
async def start_camera_service_task():
    logger.info("Starting camera service...")
    asyncio.create_task(camera_service.run_camera_loop())
    logger.info("Camera service started in background.")

@app.on_event("shutdown")
def shutdown_ngrok_tunnel():
    logger.info("Shutting down ngrok tunnel...")
    ngrok.kill()