import cv2
import numpy as np
import logging
import json
import uuid
import asyncio
from typing import Dict
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

# --- Global State for Camera Management ---
# In a real-world scenario, a more robust solution like Redis would be better.
ACTIVE_CLIENTS: Dict[str, WebSocket] = {}
LATEST_FRAMES: Dict[str, bytes] = {}

@router.get("/cameras")
async def get_cameras():
    cameras = [{"id": "local_0", "name": "Local Server Webcam", "type": "local"}]
    for client_id in ACTIVE_CLIENTS:
        cameras.append({
            "id": client_id,
            "name": f"Remote Client {client_id[:8]}",
            "type": "remote_websocket",
            "stream_ws_url": f"ws://localhost:8000/ws/view/{client_id}"
        })
    return cameras

@router.websocket("/ws/view/{source_id}")
async def view_stream_endpoint(websocket: WebSocket, source_id: str):
    await websocket.accept()
    logger.info(f"Viewer connected to watch stream from: {source_id}")
    try:
        while source_id in ACTIVE_CLIENTS:
            if source_id in LATEST_FRAMES:
                await websocket.send_bytes(LATEST_FRAMES[source_id])
                await asyncio.sleep(1/30)  # Stream at ~30 FPS
            else:
                await asyncio.sleep(0.1)
        logger.warning(f"Source {source_id} is no longer available.")
    except WebSocketDisconnect:
        logger.info(f"Viewer for stream {source_id} disconnected.")
    except Exception as e:
        logger.error(f"Error in viewer for stream {source_id}: {e}")

def get_priority_level(english_name: str) -> int:
    if english_name in HIGH_PRIORITY_OBJECTS: return 3
    elif english_name in MEDIUM_PRIORITY_OBJECTS: return 2
    else: return 1


class ConnectionManager:
    def __init__(self, detector: ObjectDetector):
        self.detector = detector
        self.previous_tracked_objects = {}
        self.previous_static_objects = {}
        self.last_processed_frame = {}

    async def handle_ws(self, websocket: WebSocket):
        await websocket.accept()

        client_id = str(uuid.uuid4())
        ACTIVE_CLIENTS[client_id] = websocket

        self.previous_tracked_objects[client_id] = set()
        self.previous_static_objects[client_id] = set()
        self.last_processed_frame[client_id] = None

        logger.info(f"Client connected with ID: {client_id}")
        last_log_text = ""
        empty_response = json.dumps({"text": "", "boxes": []})

        try:
            while True:
                data = await websocket.receive_bytes()
                LATEST_FRAMES[client_id] = data

                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_text(empty_response)
                    continue

                last_frame = self.last_processed_frame.get(client_id)
                if last_frame is not None:
                    diff = cv2.absdiff(frame, last_frame)
                    change_percentage = np.count_nonzero(diff) / frame.size
                    if change_percentage < 0.1:
                        await websocket.send_text(empty_response)
                        continue
                    elif change_percentage > 0.8:
                        self.previous_tracked_objects[client_id].clear()
                        self.previous_static_objects[client_id].clear()

                def process_frame_logic():
                    h, w, _ = frame.shape
                    analyzer = SpatialAnalyzer(frame_width=w, frame_height=h)
                    raw_detections = self.detector.detect(frame)

                    if not raw_detections:
                        self.previous_tracked_objects[client_id].clear()
                        self.previous_static_objects[client_id].clear()
                        return None

                    new_objects = []
                    current_tracked = set()
                    current_static = set()

                    for raw_det in raw_detections:
                        processed_obj = analyzer.analyze(raw_det)
                        if processed_obj.track_id is not None:
                            current_tracked.add(processed_obj.track_id)
                            if processed_obj.track_id not in self.previous_tracked_objects.get(client_id, set()):
                                new_objects.append(raw_det)
                        else:
                            current_static.add(processed_obj.name)
                            if processed_obj.name not in self.previous_static_objects.get(client_id, set()):
                                new_objects.append(raw_det)

                    self.previous_tracked_objects[client_id] = current_tracked
                    self.previous_static_objects[client_id] = current_static

                    if not new_objects: return None
                    self.last_processed_frame[client_id] = frame.copy()

                    combined_objects = []
                    for raw_det in new_objects:
                        processed_obj = analyzer.analyze(raw_det)
                        english_name = next((en for en, ru in RUSSIAN_NAMES.items() if ru == processed_obj.name),
                                            processed_obj.name)
                        priority = get_priority_level(english_name)
                        combined_objects.append(
                            {"processed": processed_obj, "confidence": raw_det.confidence, "priority": priority})

                    combined_objects.sort(key=lambda x: x["priority"], reverse=True)
                    sorted_processed_objects = [item["processed"] for item in combined_objects]
                    text_result = format_message(sorted_processed_objects)

                    return {
                        "text": text_result,
                        "boxes": [{
                            "name": item["processed"].name,
                            "xmin": round(item["processed"].normalized_box[0], 4),
                            "ymin": round(item["processed"].normalized_box[1], 4),
                            "xmax": round(item["processed"].normalized_box[2], 4),
                            "ymax": round(item["processed"].normalized_box[3], 4),
                            "distance_cm": round(item["processed"].distance_cm, 2) if item[
                                                                                          "processed"].distance_cm is not None else None,
                            "confidence": round(item["confidence"], 4), "priority": item["priority"],
                            "mask_points": [[round(p, 4) for p in point] for point in
                                            item["processed"].normalized_mask_points] if item[
                                "processed"].normalized_mask_points else None,
                            "track_id": item["processed"].track_id
                        } for item in combined_objects]
                    }

                result = await run_in_threadpool(process_frame_logic)

                if result:
                    if result["text"] != last_log_text:
                        logger.info(f"Detection for {client_id[:8]}: {result['text']}")
                        last_log_text = result["text"]
                    await websocket.send_text(json.dumps(result))
                else:
                    await websocket.send_text(empty_response)

        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected.")
        except Exception as e:
            logger.error(f"Error with client {client_id}: {e}")
        finally:
            if client_id in ACTIVE_CLIENTS: del ACTIVE_CLIENTS[client_id]
            if client_id in LATEST_FRAMES: del LATEST_FRAMES[client_id]
            if client_id in self.previous_tracked_objects: del self.previous_tracked_objects[client_id]
            if client_id in self.previous_static_objects: del self.previous_static_objects[client_id]
            if client_id in self.last_processed_frame: del self.last_processed_frame[client_id]