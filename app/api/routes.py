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
        self.previous_tracked_objects = set()
        self.previous_static_objects = set()
        self.last_processed_frame = None

    async def handle_ws(self, websocket: WebSocket):
        await websocket.accept()
        logger.info("Client connected to WebSocket.")
        last_log_text = ""
        
        # Define a standard empty response to use as a heartbeat
        empty_response = json.dumps({"text": "", "boxes": []})

        try:
            while True:
                # 1. Wait to receive a frame
                data = await websocket.receive_bytes()
                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_text(empty_response) # Send heartbeat if frame is invalid
                    continue

                # Frame comparison for smart throttling
                if self.last_processed_frame is not None:
                    diff = cv2.absdiff(frame, self.last_processed_frame)
                    non_zero_count = np.count_nonzero(diff)
                    change_percentage = non_zero_count / (frame.shape[0] * frame.shape[1])
                    
                    if change_percentage < 0.1: # 10% change threshold for static scenes
                        await websocket.send_text(empty_response) # Send heartbeat for static scenes
                        continue
                    elif change_percentage > 0.8: # 80% change threshold for scene change
                        self.previous_tracked_objects.clear()
                        self.previous_static_objects.clear()

                # Define the frame processing logic (blocking part)
                def process_frame():
                    h, w, _ = frame.shape
                    analyzer = SpatialAnalyzer(frame_width=w, frame_height=h)

                    raw_detections = self.detector.detect(frame)

                    if not raw_detections:
                        self.previous_tracked_objects.clear()
                        self.previous_static_objects.clear()
                        return None

                    current_tracked_objects = set()
                    current_static_objects = set()
                    new_objects = []

                    for raw_det in raw_detections:
                        processed_obj = analyzer.analyze(raw_det)

                        if processed_obj.track_id is not None:
                            current_tracked_objects.add(processed_obj.track_id)
                            if processed_obj.track_id not in self.previous_tracked_objects:
                                new_objects.append(raw_det)
                        else:
                            current_static_objects.add(processed_obj.name)
                            if processed_obj.name not in self.previous_static_objects:
                                new_objects.append(raw_det)

                    self.previous_tracked_objects = current_tracked_objects
                    self.previous_static_objects = current_static_objects

                    if not new_objects:
                        return None

                    self.last_processed_frame = frame.copy()

                    combined_objects = []
                    for raw_det in new_objects:
                        processed_obj = analyzer.analyze(raw_det)

                        english_name = next((en for en, ru in RUSSIAN_NAMES.items() if ru == processed_obj.name),
                                            processed_obj.name)

                        priority = get_priority_level(english_name)

                        combined_objects.append({
                            "processed": processed_obj,
                            "confidence": raw_det.confidence,
                            "priority": priority
                        })

                    combined_objects.sort(key=lambda x: x["priority"], reverse=True)

                    sorted_processed_objects = [item["processed"] for item in combined_objects]

                    text_result = format_message(sorted_processed_objects)

                    response_data = {
                        "text": text_result,
                        "boxes": [
                            {
                                "name": item["processed"].name,
                                "xmin": round(item["processed"].normalized_box[0], 4),
                                "ymin": round(item["processed"].normalized_box[1], 4),
                                "xmax": round(item["processed"].normalized_box[2], 4),
                                "ymax": round(item["processed"].normalized_box[3], 4),
                                "distance_cm": round(item["processed"].distance_cm, 2) if item[
                                                                                              "processed"].distance_cm is not None else None,
                                "confidence": round(item["confidence"], 4),
                                "priority": item["priority"],
                                "mask_points": [[round(p, 4) for p in point] for point in
                                                item["processed"].normalized_mask_points] if item[
                                    "processed"].normalized_mask_points else None,
                                "track_id": item["processed"].track_id
                            } for item in combined_objects
                        ]
                    }
                    return text_result, response_data

                # 2. Process that frame in a thread pool (blocking call offloaded)
                result = await run_in_threadpool(process_frame)

                # 3. Send the result back if processing was successful
                if result:
                    text_result, response_data = result
                    if text_result != last_log_text:
                        logger.info(f"Detection: {text_result}")
                        last_log_text = text_result

                    await websocket.send_text(json.dumps(response_data))
                else:
                    # Send a heartbeat if processing resulted in no new objects
                    await websocket.send_text(empty_response)

        except WebSocketDisconnect:
            logger.info("Client disconnected from WebSocket.")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
