import cv2
import numpy as np
import logging
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.detector import ObjectDetector
from app.domain.logic import SpatialAnalyzer
from starlette.concurrency import run_in_threadpool
from app.domain.priorities import HIGH_PRIORITY_OBJECTS, MEDIUM_PRIORITY_OBJECTS
from app.domain.message_formatter import format_message, RUSSIAN_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

router = APIRouter()

def get_priority_level(english_name: str) -> int:
    if english_name in HIGH_PRIORITY_OBJECTS:
        return 3
    elif english_name in MEDIUM_PRIORITY_OBJECTS:
        return 2
    else:
        return 1

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

                def process_frame():
                    np_arr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        return None

                    h, w, _ = frame.shape
                    analyzer = SpatialAnalyzer(frame_width=w, frame_height=h)
                    
                    raw_detections = self.detector.detect(frame)
                    
                    if not raw_detections:
                        return None

                    combined_objects = []
                    for raw_det in raw_detections:
                        processed_obj = analyzer.analyze(raw_det)
                        
                        # Get the original English name for priority checking
                        # Reverse lookup from Russian name if needed, or use name directly
                        english_name = next((en for en, ru in RUSSIAN_NAMES.items() if ru == processed_obj.name), processed_obj.name)
                        
                        priority = get_priority_level(english_name)

                        combined_objects.append({
                            "processed": processed_obj,
                            "confidence": raw_det.confidence,
                            "priority": priority
                        })

                    # Sort by priority (descending)
                    combined_objects.sort(key=lambda x: x["priority"], reverse=True)

                    sorted_processed_objects = [item["processed"] for item in combined_objects]

                    text_result = format_message(sorted_processed_objects)
                    
                    response_data = {
                        "text": text_result,
                        "boxes": [
                            {
                                "name": item["processed"].name,
                                "xmin": item["processed"].normalized_box[0],
                                "ymin": item["processed"].normalized_box[1],
                                "xmax": item["processed"].normalized_box[2],
                                "ymax": item["processed"].normalized_box[3],
                                "distance_cm": item["processed"].distance_cm,
                                "confidence": item["confidence"],
                                "priority": item["priority"] # Added priority here
                            } for item in combined_objects
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