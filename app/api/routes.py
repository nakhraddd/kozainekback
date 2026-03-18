import cv2
import numpy as np
import logging
import json
import uuid
import asyncio
import os
import sys
from typing import Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.detector import ObjectDetector, YoloDetector
from app.domain.logic import SpatialAnalyzer
from starlette.concurrency import run_in_threadpool
from app.domain.priorities import HIGH_PRIORITY_OBJECTS, MEDIUM_PRIORITY_OBJECTS
from app.domain.message_formatter import format_message, RUSSIAN_NAMES
from app.services.voice_output import VoiceAssistant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

router = APIRouter()

# --- Global State for Camera Management ---
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
                try:
                    await websocket.send_bytes(LATEST_FRAMES[source_id])
                except WebSocketDisconnect:
                    break
            else:
                await asyncio.sleep(0.1)
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
        self.voice_assistant = VoiceAssistant()
        self.previous_tracked_objects = {}
        self.previous_static_objects = {}
        self.last_processed_frame = {}
        self.last_detection_result = {}

    async def handle_ws(self, websocket: WebSocket):
        await websocket.accept()
        client_id = str(uuid.uuid4())
        ACTIVE_CLIENTS[client_id] = websocket

        self.previous_tracked_objects[client_id] = set()
        self.previous_static_objects[client_id] = set()
        self.last_processed_frame[client_id] = None
        self.last_detection_result[client_id] = {"text": "", "boxes": []}

        logger.info(f"Client connected with ID: {client_id}")
        last_log_text = ""
        
        new_frame_event = asyncio.Event()
        running = True

        async def receive_loop():
            nonlocal running
            try:
                while running:
                    data = await websocket.receive_bytes()
                    LATEST_FRAMES[client_id] = data
                    new_frame_event.set()
            except WebSocketDisconnect:
                logger.info(f"Receive loop for {client_id} disconnected.")
            finally:
                running = False
                new_frame_event.set()

        receiver_task = asyncio.create_task(receive_loop())

        try:
            while running:
                await new_frame_event.wait()
                new_frame_event.clear()

                if not running: break

                data = LATEST_FRAMES.get(client_id)
                if data is None: continue

                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None: continue

                last_frame = self.last_processed_frame.get(client_id)
                if last_frame is not None:
                    diff = cv2.absdiff(frame, last_frame)
                    change_percentage = np.count_nonzero(diff) / frame.size
                    if change_percentage < 0.1:
                        await websocket.send_text(json.dumps(self.last_detection_result[client_id]))
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
                        return {"text": "", "boxes": []}

                    newly_spoken_objects = []
                    current_tracked = set()
                    current_static = set()
                    
                    all_detected_objects = []

                    for raw_det in raw_detections:
                        processed_obj = analyzer.analyze(raw_det)
                        all_detected_objects.append({"processed": processed_obj, "raw": raw_det})

                        if processed_obj.track_id is not None:
                            current_tracked.add(processed_obj.track_id)
                            if processed_obj.track_id not in self.previous_tracked_objects.get(client_id, set()):
                                newly_spoken_objects.append(processed_obj)
                        else:
                            current_static.add(processed_obj.name)
                            if processed_obj.name not in self.previous_static_objects.get(client_id, set()):
                                newly_spoken_objects.append(processed_obj)

                    self.previous_tracked_objects[client_id] = current_tracked
                    self.previous_static_objects[client_id] = current_static
                    self.last_processed_frame[client_id] = frame.copy()

                    text_result = format_message(newly_spoken_objects)

                    response_boxes = []
                    for item in all_detected_objects:
                        proc_obj = item["processed"]
                        raw_obj = item["raw"]
                        english_name = next((en for en, ru in RUSSIAN_NAMES.items() if ru == proc_obj.name), proc_obj.name)
                        priority = get_priority_level(english_name)
                        
                        response_boxes.append({
                            "name": proc_obj.name,
                            "xmin": round(proc_obj.normalized_box[0], 4), "ymin": round(proc_obj.normalized_box[1], 4),
                            "xmax": round(proc_obj.normalized_box[2], 4), "ymax": round(proc_obj.normalized_box[3], 4),
                            "distance_cm": round(proc_obj.distance_cm, 2) if proc_obj.distance_cm is not None else None,
                            "confidence": round(raw_obj.confidence, 4),
                            "priority": priority,
                            "mask_points": [[round(p, 4) for p in point] for point in proc_obj.normalized_mask_points] if proc_obj.normalized_mask_points else None,
                            "track_id": proc_obj.track_id
                        })

                    return {"text": text_result, "boxes": response_boxes}

                result = await run_in_threadpool(process_frame_logic)

                if result:
                    self.last_detection_result[client_id] = result
                    if result["text"] and result["text"] != last_log_text:
                        logger.info(f"Speaking for {client_id[:8]}: {result['text']}")
                        await self.voice_assistant.speak(result["text"])
                        last_log_text = result["text"]
                    await websocket.send_text(json.dumps(result))
                else:
                    await websocket.send_text(json.dumps(self.last_detection_result[client_id]))

        except Exception as e:
            logger.error(f"Error in processor loop for {client_id}: {e}", exc_info=True)
        finally:
            running = False
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass

            if client_id in ACTIVE_CLIENTS: del ACTIVE_CLIENTS[client_id]
            if client_id in LATEST_FRAMES: del LATEST_FRAMES[client_id]
            if client_id in self.previous_tracked_objects: del self.previous_tracked_objects[client_id]
            if client_id in self.previous_static_objects: del self.previous_static_objects[client_id]
            if client_id in self.last_processed_frame: del self.last_processed_frame[client_id]
            if client_id in self.last_detection_result: del self.last_detection_result[client_id]
            logger.info(f"Client {client_id} disconnected and cleaned up.")

# Determine the base path for bundled files in PyInstaller
if getattr(sys, 'frozen', False):
    # Running in a PyInstaller bundle
    bundle_dir = sys._MEIPASS
else:
    # Running in a normal Python environment
    # Assume the model is in the project root, which is two levels up from app/api/routes.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    bundle_dir = project_root

model_file_name = "yolov8n-seg.pt"
model_path_to_use = os.path.join(bundle_dir, model_file_name)

detector = YoloDetector(model_path=model_path_to_use)
manager = ConnectionManager(detector=detector)
