from fastapi import FastAPI, WebSocket
from app.api.routes import router, ConnectionManager
from app.services.detector import YoloDetector

app = FastAPI(title="KOZAINEK API")

# Upgrading to yolov8s.pt for better accuracy
detector = YoloDetector(model_path="yolov8s-seg.pt")
manager = ConnectionManager(detector=detector)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.handle_ws(websocket)

app.include_router(router)