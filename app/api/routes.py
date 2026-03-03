import cv2
import numpy as np
from fastapi import APIRouter, WebSocket
from app.services.detector import ObjectDetector
from app.domain.logic import SpatialAnalyzer

router = APIRouter()

class ConnectionManager:
    def __init__(self, detector: ObjectDetector):
        self.detector = detector

    async def handle_ws(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_bytes()
                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    h, w, _ = frame.shape
                    analyzer = SpatialAnalyzer(frame_width=w, frame_height=h)
                    
                    raw_detections = self.detector.detect(frame)
                    
                    processed = [analyzer.analyze(d) for d in raw_detections]
                    
                    if processed:
                        text_result = ", ".join([f"{p.name} {p.distance} {p.position}" for p in processed])
                        await websocket.send_text(text_result)
        except Exception:
            pass