from fastapi import FastAPI, WebSocket
from app.api.routes import router, ConnectionManager
from app.services.detector import YoloDetector
from app.services.camera_service import CameraService # Import CameraService
import logging
from pyngrok import ngrok
from starlette.concurrency import run_in_threadpool
import asyncio # Import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="KOZAINEK API")

# Upgrading to yolov8s.pt for better accuracy
detector = YoloDetector(model_path="yolov8s-seg.pt")
manager = ConnectionManager(detector=detector)

# Initialize CameraService
camera_service = CameraService()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.handle_ws(websocket)

app.include_router(router)

@app.on_event("startup")
async def start_ngrok_tunnel():
    try:
        # You might need to set your ngrok auth token if you haven't already
        # ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
        
        tunnel = await run_in_threadpool(ngrok.connect, 8000, domain="anemone-intimate-utterly.ngrok-free.app")
        logger.info(f"Ngrok tunnel started at: {tunnel.public_url}")
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}")

@app.on_event("startup")
async def start_camera_service():
    logger.info("Starting camera service...")
    # Run the camera service in a separate task so it doesn't block the main event loop
    asyncio.create_task(camera_service.start_camera())
    logger.info("Camera service started in background.")

@app.on_event("shutdown")
def shutdown_ngrok_tunnel():
    logger.info("Shutting down ngrok tunnel...")
    ngrok.kill()
    # Optionally, add camera service shutdown logic here if needed
    # camera_service.video_capture.release()
    # cv2.destroyAllWindows()
