import cv2
import numpy as np
import logging
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.detector import ObjectDetector
from app.domain.logic import SpatialAnalyzer
from starlette.concurrency import run_in_threadpool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    def __init__(self, detector: ObjectDetector):
        self.detector = detector

    async def handle_ws(self, websocket: WebSocket):
        await websocket.accept()
        logger.info("Client connected to WebSocket.")
        last_log_text = ""
        
        try:
            while True:
                data = await websocket.receive_bytes()

                # Run the CPU-bound processing in a thread pool
                def process_frame():
                    np_arr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        return None

                    h, w, _ = frame.shape
                    analyzer = SpatialAnalyzer(frame_width=w, frame_height=h)
                    
                    raw_detections = self.detector.detect(frame)
                    processed = [analyzer.analyze(d) for d in raw_detections]
                    
                    if not processed:
                        return None

                    text_result_parts = []
                    for p in processed:
                        distance_info = f"({p.distance_cm:.2f}cm)" if p.distance_cm is not None else "(unknown distance)"
                        text_result_parts.append(f"{p.name} {p.distance} {p.position} {distance_info}")
                    text_result = ", ".join(text_result_parts)
                    
                    response_data = {
                        "text": text_result,
                        "boxes": [
                            {
                                "name": p.name,
                                "xmin": p.normalized_box[0],
                                "ymin": p.normalized_box[1],
                                "xmax": p.normalized_box[2],
                                "ymax": p.normalized_box[3],
                                "distance_cm": p.distance_cm
                            } for p in processed
                        ]
                    }
                    return text_result, response_data

                result = await run_in_threadpool(process_frame)

                if result:
                    text_result, response_data = result
                    if text_result != last_log_text:
                        logger.info(f"Detection: {text_result}")
                        last_log_text = text_result
                    
                    await websocket.send_text(json.dumps(response_data))

        except WebSocketDisconnect:
            logger.info("Client disconnected from WebSocket.")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")